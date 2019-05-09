import mpi4py.MPI as mpi
import numpy as np
import compyle.array as carr
from compyle.array import get_backend
from compyle.api import annotate
from compyle.parallel import Scan
from pyzoltan.hetero.comm import Comm, dtype_to_mpi


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
    def __init__(self, ndims, dtype, coords=[], gids=None, weights=None,
                 proc_weights=None, root=0, tag=111, backend=None):
        self.comm = mpi.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.root = root
        self.backend = get_backend(backend)
        self.coords = coords
        self.gids = gids
        self.ndims = ndims
        self.weights = weights
        self.proc_weights = proc_weights
        self.out_list = None
        self.tag = tag
        self.dtype = dtype
        self.procs = None
        self.plan_recvs = []
        self.plan_transfers = []

        if self.gids and self.gids.dtype != np.int32:
            raise TypeError("gids should be of type int32 and not %s" % \
                            self.gids.dtype)

        if self.rank == self.root:
            self.procs = carr.arange(0, self.size, 1, dtype=np.int32,
                                     backend=self.backend)

        if self.rank == self.root and self.gids is None:
            nobjs = self.coords[0].length
            self.gids = carr.arange(0, nobjs, 1, dtype=np.int32,
                                    backend=self.backend)


        if self.rank == self.root and self.weights is None:
            ndata = self.gids.length
            self.weights = carr.empty(ndata, np.float32,
                                      backend=self.backend)
            self.weights.fill(1. / ndata)

        if self.rank == self.root and self.proc_weights is None:
            self.proc_weights = carr.empty(self.size, np.float32,
                                           backend=self.backend)
            self.proc_weights.fill(1. / self.size)

        # views of original arrays
        self.procs_view = self.procs.get_view() if self.procs else None
        self.proc_weights_view = self.proc_weights.get_view() \
                if self.proc_weights else None
        self.weights_view = self.weights.get_view() if self.weights else None
        self.gids_view = self.gids.get_view() \
                if self.gids else None
        self.coords_view = [x.get_view() for x in self.coords] if self.coords else []

        self.max = np.zeros(self.ndims, dtype=self.dtype)
        self.min = np.zeros(self.ndims, dtype=self.dtype)

        for i, x in enumerate(self.coords_view):
            x.update_min_max()
            self.max[i] = x.maximum
            self.min[i] = x.minimum

    def _partition_procs(self):
        # NOTE: This can be made better using the weights
        # of the procs and communication costs
        part_idx = self.procs_view.length // 2
        target_w = carr.sum(self.proc_weights_view[:part_idx])
        return part_idx, target_w

    def _partition_domain(self, target_w):
        lengths = self.max - self.min

        maxlen_idx = np.argmax(lengths)

        # sort data and gids according to max length dim

        order = carr.argsort(self.coords_view[maxlen_idx])

        inp_list = [self.gids_view] + self.coords_view

        # FIXME: Is allocating more expensive than memcopy?
        if not self.out_list:
            self.out_list = [
                    carr.empty(order.length, ary.dtype, backend=self.backend)
                    for ary in inp_list
                    ]
        else:
            for out in self.out_list:
                out.resize(order.length)

        carr.align(inp_list, order, out_list=self.out_list,
                   backend=self.backend)
        self.gids_view = self.out_list[0].copy()
        for i, out in enumerate(self.out_list[1:]):
            self.coords_view[i] = out.copy()

        target_idx = carr.zeros(1, dtype=np.int32, backend=self.backend)

        split_idx_knl = Scan(inp_partition_domain, out_partition_domain,
                             'a+b', dtype=self.weights_view.dtype,
                             backend=self.backend)

        split_idx_knl(weights=self.weights_view, target_w=target_w,
                      target_idx=target_idx)

        target_idx = int(target_idx[0])

        return maxlen_idx, target_idx

    def gather(self):
        nobjs_to_get = None
        if self.rank == self.root:
            nobjs_to_get = np.zeros(self.size,
                                    dtype=self.gids_view.dtype)

        nobjs = self.gids_view.length

        self.comm.Gather(np.array(nobjs, dtype=np.int32),
                         nobjs_to_get, root=self.root)

        if self.rank == self.root:
            total_nobjs = np.sum(nobjs_to_get)
            gath_gids = carr.empty(total_nobjs, np.int32)
            gath_weights = carr.empty(total_nobjs, np.float32)
            gath_coords = [carr.empty(total_nobjs, self.dtype) \
                    for i in range(self.ndims)]
            gath_proc_weights = carr.empty(self.size, np.float32)
        else:
            gath_gids, gath_weights, gath_proc_weights = None, None, None
            gath_coords = [None] * self.ndims

        self.comm.Gatherv(sendbuf=[self.gids_view.get_buff(),
                                   dtype_to_mpi(self.gids_view.dtype)],
                          recvbuf=[gath_gids.get_buff(), nobjs_to_get],
                          root=self.root)

        self.comm.Gatherv(sendbuf=[self.weights_view.get_buff(),
                                   dtype_to_mpi(self.weights_view.dtype)],
                          recvbuf=[gath_weights.get_buff(), nobjs_to_get],
                          root=self.root)

        for i, x in enumerate(self.coords_view):
            self.comm.Gatherv(sendbuf=[x.get_buff(),
                                       dtype_to_mpi(x.dtype)],
                              recvbuf=[gath_coords[i].get_buff(), nobjs_to_get],
                              root=self.root)

        self.comm.Gather([self.proc_weights_view.get_buff(),
                          dtype_to_mpi(self.proc_weights_view.dtype)],
                          gath_proc_weights.get_buff(),
                          root=self.root)

        self.gids_view = gath_gids
        self.weights_view = gath_weights
        self.coords_view = gath_coords
        self.proc_weights_view = gath_proc_weights

        self.all_exec_times = np.zeros(self.size, dtype=np.float32)
        self.all_exec_nobjs = np.zeros(self.size, dtype=np.int32)

    def adjust_proc_weights(self, exec_time, exec_nobjs, exec_count):
        # set new proc weights based on exec times
        # send all times to root
        scale_fac = self.total_num_objs * (exec_time / exec_nobjs)

        self.lb.comm.Gather(np.array(exec_time, dtype=np.float32),
                            self.all_exec_times, root=self.root)

        self.lb.comm.Gather(np.array(exec_nobjs, dtype=np.int32),
                            self.all_exec_nobjs, root=self.root)

        if self.rank == self.root:
            eff_weights = self.all_exec_nobjs / \
                    (self.total_num_objs * exec_count)
            scale_fac = 1. / np.sum(eff_weights / self.all_exec_times)
            proc_weights = scale_fac * eff_weights / self.all_exec_times

            self.proc_weights = wrap(proc_weights, backend=self.backend)
            self.proc_weights_view = self.proc_weights.get_view()

    def migrate_objects(self):
        pass

    def load_balance(self):
        # FIXME: These tags might mess with other sends and receives
        reqs_coords = []
        req_objids = None
        req_w = None
        req_max, req_min = None, None
        if self.rank != self.root:
            status = mpi.Status()
            nrecv_coords = np.empty(1, dtype=np.int32)
            nrecv_procs = np.empty(1, dtype=np.int32)

            req_nprocs = self.comm.Irecv(
                    nrecv_procs, source=mpi.ANY_SOURCE, tag=self.tag + 1
                    )

            req_nprocs.Wait(status=status)
            self.parent = status.Get_source()

            req_ncoords = self.comm.Irecv(
                    nrecv_coords, source=self.parent, tag=self.tag + 2
                    )

            self.nrecv_procs = int(nrecv_procs)
            self.procs_view = carr.empty(self.nrecv_procs, np.int32,
                                    backend=self.backend)

            req_procs = self.comm.Irecv(self.procs_view.get_buff(),
                                        source=self.parent, tag=self.tag + 3)

            req_ncoords.Wait()

            self.nrecv_coords = int(nrecv_coords)

            self.gids_view = carr.empty(self.nrecv_coords, np.int32,
                                         backend=self.backend)

            req_objids = self.comm.Irecv(self.gids_view.get_buff(),
                                         source=self.parent, tag=self.tag + 4)

            self.proc_weights_view = carr.empty(self.nrecv_procs, self.dtype,
                                           backend=self.backend)

            req_procw = self.comm.Irecv(self.proc_weights_view.get_buff(),
                                        source=self.parent, tag=self.tag + 5)

            self.weights_view = carr.empty(self.nrecv_coords, self.dtype,
                                      backend=self.backend)

            req_w = self.comm.Irecv(self.weights_view.get_buff(),
                                    source=self.parent, tag=self.tag + 6)

            for i in range(self.ndims):
                x = carr.empty(self.nrecv_coords, self.dtype,
                               backend=self.backend)
                reqs_coords.append(self.comm.Irecv(x.get_buff(),
                                 source=self.parent, tag=self.tag + 7 + i))
                self.coords_view.append(x)

            req_max = self.comm.Irecv(self.max, source=self.parent,
                                      tag=self.tag + 10)
            req_min = self.comm.Irecv(self.min, source=self.parent,
                                      tag=self.tag + 11)

            req_procs.Wait()
            req_procw.Wait()

            if self.procs_view.length == 1:
                mpi.Request.Waitall(reqs_coords)
                req_objids.Wait()
                self.num_objs = self.gids_view.length
                req_max.Wait()
                req_min.Wait()
                req_w.Wait()
                self.make_comm_plan()
                return

        while self.procs_view.length != 1:
            part_idx, target_w = self._partition_procs()

            right_proc = int(self.procs_view[part_idx])

            nsend_rprocs = np.asarray(self.procs_view.length - part_idx,
                                      dtype=np.int32)

            # transfer nsend procs
            self.comm.Send(nsend_rprocs, dest=right_proc, tag=self.tag + 1)

            if reqs_coords:
                mpi.Request.Waitall(reqs_coords)
                reqs_coords = []

            if req_max:
                req_max.Wait()
                req_max = None

            if req_min:
                req_min.Wait()
                req_min = None

            if req_w:
                req_w.Wait()
                req_w = None

            if req_objids:
                req_objids.Wait()
                req_objids = None

            right_max = self.max.copy()
            right_min = self.min.copy()

            maxlen_idx, target_idx = self._partition_domain(target_w)

            part_dim = self.coords_view[maxlen_idx]
            new_max = part_dim[target_idx - 1]
            new_min = part_dim[target_idx]

            self.max[maxlen_idx] = new_max
            right_min[maxlen_idx] = new_min

            nsend_rcoords = np.asarray(self.coords_view[0].length - target_idx,
                                     dtype=np.int32)

            # transfer nsend data
            self.comm.Send(nsend_rcoords, dest=right_proc, tag=self.tag + 2)

            # transfer procs
            self.comm.Send(self.procs_view.get_buff(offset=part_idx),
                           dest=right_proc, tag=self.tag + 3)

            # transfer proc weights
            self.comm.Send(self.proc_weights_view.get_buff(offset=part_idx),
                           dest=right_proc, tag=self.tag + 5)

            # transfer x, y, z
            for i, x in enumerate(self.coords_view):
                self.comm.Send(x.get_buff(offset=target_idx),
                               dest=right_proc, tag=self.tag + 7 + i)

            # transfer data weights
            self.comm.Send(self.weights_view.get_buff(offset=target_idx),
                           dest=right_proc, tag=self.tag + 6)

            # transfer obj ids
            self.comm.Send(self.gids_view.get_buff(offset=target_idx),
                           dest=right_proc, tag=self.tag + 4)

            self.comm.Send(right_max, dest=right_proc, tag=self.tag + 10)

            self.comm.Send(right_min, dest=right_proc, tag=self.tag + 11)

            self.procs_view.resize(part_idx)
            self.proc_weights_view.resize(part_idx)
            self.gids_view.resize(target_idx)
            for x in self.coords_view:
                x.resize(target_idx)
            self.weights_view.resize(target_idx)

        self.num_objs = self.gids_view.length
        self.make_comm_plan()

    def point_assign(self, point):
        for proc in self.size:
            proc_max = self.maxs[proc * self.ndims: (proc + 1) * self.ndims]
            proc_min = self.mins[proc * self.ndims: (proc + 1) * self.ndims]

            if np.all(point <= proc_max) and np.all(point >= proc_min):
                return proc

    def make_comm_plan(self):
        # Send object ids back to root
        self.nobjs_per_proc = None
        if self.rank == self.root:
            self.nobjs_per_proc = np.zeros(self.size, dtype=self.gids_view.dtype)

        self.comm.Gather(np.array(self.num_objs, dtype=np.int32),
                         self.nobjs_per_proc, root=self.root)

        if self.rank == self.root:
            self.total_num_objs = np.sum(self.nobjs_per_proc,
                                         dtype=self.gids_view.dtype)
            self.all_gids = np.empty(self.total_num_objs, np.int32)
        else:
            self.total_num_objs = np.array(0, dtype=self.gids_view.dtype)
            self.all_gids = None

        self.comm.Bcast(self.total_num_objs, root=self.root)

        self.maxs = carr.empty(self.ndims * self.size, self.dtype,
                               backend=self.backend)
        self.mins = carr.empty(self.ndims * self.size, self.dtype,
                               backend=self.backend)

        self.comm.Allgather(self.max, self.maxs.get_buff())
        self.comm.Allgather(self.min, self.mins.get_buff())

        self.comm.Gatherv(sendbuf=[self.gids_view.get_buff(),
                                   dtype_to_mpi(self.gids_view.dtype)],
                          recvbuf=[self.all_gids, self.nobjs_per_proc],
                          root=self.root)

        proclist = []
        if self.rank == self.root:
            procstart_ids = np.zeros(self.size, dtype=np.int32)

            np.cumsum(self.nobjs_per_proc[:-1], out=procstart_ids[1:])

            proclist = np.empty(self.total_num_objs, dtype=np.int32)

            for proc in range(self.size):
                start = procstart_ids[proc]
                proclist[start:start + self.nobjs_per_proc[proc]] = proc

        # FIXME: tag
        self.plan = Comm(proclist, sorted=True, root=self.root, tag=self.tag)

    def comm_do_post(self, senddata, recvdata):
        self.plan_recvs.append(recvdata)
        self.plan.comm_do_post(senddata, recvdata)

    def comm_do_wait(self):
        self.plan.comm_do_wait()

    def comm_do(self, senddata, recvdata):
        self.plan.comm_do(senddata, recvdata)
