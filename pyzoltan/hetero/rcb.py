import mpi4py.MPI as mpi
import numpy as np
import compyle.array as carr
from compyle.array import get_backend
from compyle.api import annotate
from compyle.parallel import Scan


def dbg_print(msg):
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    print("Rank %s: %s" % (rank, msg))


@annotate
def inp_partition_domain(i, weights):
    return weights[i]


@annotate
def out_partition_domain(i, item, prev_item, target_w, target_idx):
    if item >= target_w and prev_item < target_w:
        target_idx[0] = i


class RCB(object):
    '''
    Recursive coordinate bisection
    ------------------------------

    The list of processes is bisected into two sets. The total
    weight in each set is calculated and called target weight.
    The objects are sorted based on the coordinate of the
    longest dimension. The object weights are summed until the
    target weight is reached. Thus, we have two partitions
    of objects as well as processes. This is continued recursively
    until each set has a single process.
    '''
    def __init__(self, ndims, dtype, data=[], object_ids=None, weights=None,
                 proc_weights=None, root=0, tag=111, backend=None):
        self.comm = mpi.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.root = root
        self.size = self.comm.Get_size()
        self.backend = get_backend(backend)
        self.data = data
        self.object_ids = object_ids
        self.ndims = ndims
        self.weights = weights
        self.proc_weights = proc_weights
        self.out_list = None
        self.tag = tag
        self.dtype = dtype

        self.procs = None
        if self.rank == self.root:
            self.procs = carr.arange(0, self.size, 1, dtype=np.int32,
                                     backend=self.backend)

        if self.rank == self.root and self.weights is None:
            ndata = self.object_ids.length
            self.weights = carr.empty(ndata, np.float32,
                                      backend=self.backend)
            self.weights.fill(1. / ndata)

        if self.rank == self.root and self.proc_weights is None:
            self.proc_weights = carr.empty(self.size, np.float32,
                                           backend=self.backend)
            self.proc_weights.fill(1. / self.size)

    def _partition_procs(self):
        # NOTE: This can be made better using the weights
        # of the procs and communication costs
        part_idx = self.procs.length // 2
        target_w = carr.sum(self.proc_weights[:part_idx])
        return part_idx, target_w

    def _partition_domain(self, target_w):
        for x in self.data:
            x.update_min_max()

        lengths = [float(x.maximum - x.minimum) for x in self.data]

        maxlen_idx = np.argmax(lengths)

        # sort data and object_ids according to max length dim

        order = carr.argsort(self.data[maxlen_idx])

        inp_list = [self.object_ids] + self.data

        # FIXME: Is allocating more expensive than memcopy?
        if not self.out_list:
            self.out_list = [
                    carr.empty(order.length, ary.dtype, backend=self.backend)
                    for ary in inp_list
                    ]
        else:
            for out in self.out_list:
                out.resize(order.length)

        carr.align(inp_list, order, out_list=self.out_list)
        self.object_ids = self.out_list[0].copy()
        for i, out in enumerate(self.out_list[1:]):
            self.data[i] = out.copy()

        target_idx = carr.zeros(1, dtype=np.int32, backend=self.backend)

        split_idx_knl = Scan(inp_partition_domain, out_partition_domain,
                             'a+b', dtype=self.weights.dtype,
                             backend=self.backend)

        split_idx_knl(weights=self.weights, target_w=target_w,
                      target_idx=target_idx)

        return int(target_idx[0])

    def load_balance(self):
        reqs_data = []
        req_objids = None
        req_w = None
        if self.rank != self.root:
            status = mpi.Status()
            nrecv_data = np.empty(1, dtype=np.int32)
            nrecv_procs = np.empty(1, dtype=np.int32)

            req_nprocs = self.comm.Irecv(
                    nrecv_procs, source=mpi.ANY_SOURCE, tag=self.tag + 1
                    )

            req_nprocs.Wait(status=status)
            self.parent = status.Get_source()

            req_ndata = self.comm.Irecv(
                    nrecv_data, source=self.parent, tag=self.tag + 2
                    )

            self.nrecv_procs = int(nrecv_procs)
            self.procs = carr.empty(self.nrecv_procs, np.int32,
                                    backend=self.backend)

            req_procs = self.comm.Irecv(self.procs.get_buff(),
                                        source=self.parent, tag=self.tag + 3)

            req_ndata.Wait()

            self.nrecv_data = int(nrecv_data)

            self.object_ids = carr.empty(self.nrecv_data, np.int32,
                                         backend=self.backend)

            req_objids = self.comm.Irecv(self.object_ids.get_buff(),
                                         source=self.parent, tag=self.tag + 4)

            self.proc_weights = carr.empty(self.nrecv_procs, self.dtype,
                                           backend=self.backend)

            req_procw = self.comm.Irecv(self.proc_weights.get_buff(),
                                        source=self.parent, tag=self.tag + 5)

            self.weights = carr.empty(self.nrecv_data, self.dtype,
                                      backend=self.backend)

            req_w = self.comm.Irecv(self.weights.get_buff(),
                                    source=self.parent, tag=self.tag + 6)

            for i in range(self.ndims):
                x = carr.empty(self.nrecv_data, self.dtype,
                               backend=self.backend)
                reqs_data.append(self.comm.Irecv(x.get_buff(),
                                 source=self.parent, tag=self.tag + 7 + i))
                self.data.append(x)

            req_procs.Wait()
            req_procw.Wait()

            if self.procs.length == 1:
                mpi.Request.Waitall(reqs_data)
                req_objids.Wait()
                self.num_objs = self.object_ids.length
                dbg_print("Num objects = %s" % self.num_objs)
                req_w.Wait()
                return

        while self.procs.length != 1:
            part_idx, target_w = self._partition_procs()

            right_proc = int(self.procs[part_idx])

            nsend_rprocs = np.asarray(self.procs.length - part_idx,
                                      dtype=np.int32)

            # transfer nsend procs
            self.comm.Send(nsend_rprocs, dest=right_proc, tag=self.tag + 1)

            if reqs_data:
                mpi.Request.Waitall(reqs_data)

            if req_w:
                req_w.Wait()

            target_idx = self._partition_domain(target_w)

            nsend_rdata = np.asarray(self.data[0].length - target_idx,
                                     dtype=np.int32)

            # transfer nsend data
            self.comm.Send(nsend_rdata, dest=right_proc, tag=self.tag + 2)

            # transfer procs
            self.comm.Send(self.procs.get_buff(offset=part_idx),
                           dest=right_proc, tag=self.tag + 3)

            # transfer proc weights
            self.comm.Send(self.proc_weights.get_buff(offset=part_idx),
                           dest=right_proc, tag=self.tag + 5)

            # transfer x, y, z
            for i, x in enumerate(self.data):
                self.comm.Send(x.get_buff(offset=target_idx),
                               dest=right_proc, tag=self.tag + 7 + i)

            # transfer data weights
            self.comm.Send(self.weights.get_buff(offset=target_idx),
                           dest=right_proc, tag=self.tag + 6)

            if req_objids:
                req_objids.Wait()

            # transfer obj ids
            self.comm.Send(self.object_ids.get_buff(offset=target_idx),
                           dest=right_proc, tag=self.tag + 4)

            self.procs.resize(part_idx)
            self.object_ids.resize(target_idx)
            for x in self.data:
                x.resize(target_idx)

        self.num_objs = self.object_ids.length
        dbg_print("Num objects = %s" % self.num_objs)

    def make_comm_plan(self):
        # Send object ids back to root
        self.nelements = None
        if self.rank == self.root:
            self.nelements = np.zeros(self.size, dtype=np.int32)
            self.lb_objs = np.zeros(self.num_objs, np.int32)

        self.comm.Gather(np.array(self.num_objs, dtype=np.int32),
                         self.nelements, root=self.root)

        self.comm.Gatherv(sendbuf=self.object_ids.get_buff(),
                          recvbuf=(self.lb_objs.get_buff(), self.nelements),
                          root=self.root)

        proclist = None
        if self.rank == self.root:
            procstart_ids = np.zeros(self.size, dtype=np.int32)

            np.cumsum(self.nelements[:-1], out=procstart_ids[1:])

            proclist = np.empty(self.num_objs, dtype=np.int32)

            for proc in range(self.size):
                start = procstart_ids[proc]
                proclist[start:start + self.nelements[proc]] = proc

        # FIXME: tag
        self.plan = Comm(proclist, sorted=True, root=self.root, tag=self.tag)

    def comm_do_post(self, senddata, recvdata):
        self.plan.comm_do_post(senddata, recvdata)

    def comm_do_wait(self):
        self.plan.comm_do_wait()

    def comm_do(self, senddata, recvdata):
        self.plan.comm_do(senddata, recvdata)
