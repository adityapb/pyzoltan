import mpi4py.MPI as mpi
import numpy as np
import compyle.array as carr
from compyle.array import get_backend
from compyle.types import annotate, NP_TYPE_LIST
from compyle.parallel import Elementwise, Reduction, Scan
from pytools import memoize


def dbg_print(msg):
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    print("Rank %s: %s" % (rank, msg))


def dtype_to_mpi(t):
    if hasattr(mpi, '_typedict'):
        mpi_type = mpi._typedict[np.dtype(t).char]
    elif hasattr(mpi, '__TypeDict__'):
        mpi_type = mpi.__TypeDict__[np.dtype(t).char]
    else:
        raise ValueError('cannot convert type')
    return mpi_type


class CommBase(object):
    def __init__(self, proclist, tag=0, root=0, backend=None, sorted=False):
        self.comm = mpi.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.sorted = sorted
        self.backend = get_backend(backend)
        self.root = root
        if proclist:
            self.nsends = len(proclist)
        self.proclist = proclist
        self.tag = tag
        self.blocked_senddata = None
        self.requests = []

    def set_send_info(self, procs_to, lengths_to):
        self.nblocks = len(procs_to)
        ord_procs_to = np.argsort(procs_to)
        self.procs_to = procs_to[ord_procs_to]
        self.lengths_to = lengths_to[ord_procs_to]
        self.starts = np.zeros_like(self.lengths_to)
        np.cumsum(self.lengths_to[:-1], out=self.starts[1:])
        self.nsends = np.sum(self.lengths_to)

    def set_recv_info(self, procs_from, lengths_from):
        self.nrecvs = len(procs_from)
        ord_procs_from = np.argsort(procs_from)
        self.procs_from = procs_from[ord_procs_from]
        self.lengths_from = lengths_from[ord_procs_from]
        self.start_from = np.zeros_like(self.lengths_from)
        np.cumsum(self.lengths_from[:-1], out=self.start_from[1:])
        self.nreturn = np.sum(self.lengths_from)

    def get_recvdtype(self, senddtype):
        recvdtype = -1 + np.zeros_like(self.procs_from)

        if senddtype in NP_TYPE_LIST:
            senddtype = NP_TYPE_LIST.index(senddtype)
        else:
            senddtype = -1

        requests = [None for i in range(self.nrecvs)]

        for i in range(self.nrecvs):
            requests[i] = self.comm.Irecv(
                    recvdtype[i:i+1], source=self.procs_from[i],
                    tag=self.tag
                    )

        for i in range(self.nblocks):
            self.comm.Send(np.array(senddtype, dtype=np.int32),
                           dest=self.procs_to[i], tag=self.tag)

        mpi.Request.Waitall(requests)

        if not np.all(recvdtype == recvdtype[0]):
            raise ValueError("All sends should have same datatype")

        if recvdtype[0] == -1:
            return -1
        else:
            return NP_TYPE_LIST[recvdtype[0]]

    def bcast(self, senddata):
        self.comm.Bcast(senddata, root=self.root)

    def comm_do_post(self, senddata, recvdata):
        # senddata and recvdata must be compyle Arrays
        #
        # NOTE: Make another method for multiple senddata's to
        # save aligning cost

        # Make continuous buffers for each process the data
        # has to be sent to
        # store sendbuff
        if not self.sorted:
            if senddata and (self.blocked_senddata is None or \
                    self.dtype != senddata.dtype):
                self.blocked_senddata = carr.zeros(self.nsends, senddata.dtype,
                                                   backend=self.backend)
                self.dtype = senddata.dtype

            senddata.align(self.order, out=self.blocked_senddata)
        else:
            self.blocked_senddata = senddata

        for i in range(self.nrecvs):
            start = self.start_from[i]
            length = self.lengths_from[i]
            if length:
                recvbuff = recvdata.get_buff(offset=start,
                                             length=length)
                self.requests.append(self.comm.Irecv(
                        [recvbuff, dtype_to_mpi(recvdata.dtype)],
                        source=self.procs_from[i], tag=self.tag
                        ))

        self.comm.Barrier()

        for i in range(self.nblocks):
            start = self.starts[i]
            length = self.lengths_to[i]
            if length:
                sendbuff = self.blocked_senddata.get_buff(offset=start,
                                                          length=length)
                self.comm.Rsend([sendbuff, dtype_to_mpi(senddata.dtype)],
                                dest=self.procs_to[i], tag=self.tag)

    def comm_do_wait(self):
        if self.requests:
            mpi.Request.Waitall(self.requests)
            self.requests = []

    def comm_do(self, senddata, recvdata):
        self.comm_do_post(senddata, recvdata)
        self.comm_do_wait()


@annotate
def inp_unique_procs(i, proclist):
    return 1 if i == 0 or proclist[i] != proclist[i - 1] else 0


@annotate
def out_unique_procs(i, item, prev_item, proclist, procs_to, starts):
    if item != prev_item:
        procs_to[item - 1] = proclist[i]
        starts[item - 1] = i


@annotate
def find_lengths_from_starts(i, starts, lengths_to, num_procs, num_objs):
    start = starts[i]
    if i == num_procs - 1:
        end = num_objs
    else:
        end = starts[i + 1]

    lengths_to[i] = end - start


@memoize(key=lambda *args: tuple(args))
def get_elwise(f, backend):
    return Elementwise(f, backend=backend)


@memoize(key=lambda *args: tuple(args))
def get_scan(inp_f, out_f, dtype, backend):
    return Scan(input=inp_f, output=out_f, dtype=dtype,
                backend=backend)


class Comm(CommBase):
    def __init__(self, proclist, tag=0, root=0, backend=None, sorted=False):
        super(Comm, self). __init__(proclist, tag=tag, root=root,
                                    backend=backend, sorted=sorted)
        if len(self.proclist) > 0:
            if isinstance(self.proclist, np.ndarray):
                self.proclist = wrap_array(self.proclist,
                                           backend=self.backend)
            if not self.sorted:
                self.order = self._sort_proclist()

            num_procs_knl = Reduction(
                    'a+b', dtype_out=np.int32, map_func=inp_unique_procs,
                    backend=self.backend
                    )

            self.nblocks = int(num_procs_knl(self.proclist))

            starts = carr.empty(self.nblocks, np.int32, backend=self.backend)
            procs_to = carr.empty(self.nblocks, np.int32, backend=self.backend)
            lengths_to = carr.empty(self.nblocks, np.int32, backend=self.backend)

            unique_procs_knl = get_scan(inp_unique_procs, out_unique_procs,
                                        np.int32, self.backend)

            unique_procs_knl(proclist=self.proclist, procs_to=procs_to,
                             starts=starts)

            find_lengths_from_starts_knl = get_elwise(find_lengths_from_starts,
                                                     backend=self.backend)

            find_lengths_from_starts_knl(starts, lengths_to, self.nblocks,
                                         len(proclist))

            self.starts = np.array(starts.get(), dtype=np.int32)
            self.procs_to = np.array(procs_to.get(), dtype=np.int32)
            self.lengths_to = np.array(lengths_to.get(), dtype=np.int32)
        else:
            self.starts = np.empty(0, dtype=np.int32)
            self.procs_to = np.empty(0, dtype=np.int32)
            self.lengths_to = np.empty(0, dtype=np.int32)
            self.nblocks = 0

        self._comm_invert_map(tag, self.lengths_to, self.procs_to)

    def _sort_proclist(self):
        return carr.argsort(self.proclist)

    def _comm_invert_map(self, tag, lengths_to, procs_to):
        status = mpi.Status()
        msg_count = np.zeros(self.size, dtype=np.int32)
        counts = np.zeros_like(msg_count)
        nrecvs = np.zeros(1, dtype=np.int32)

        for proc in procs_to:
            msg_count[proc] = 1

        self.comm.Reduce(msg_count, counts, root=self.root)
        self.comm.Scatter(counts, nrecvs, root=self.root)
        self.nrecvs = int(nrecvs)

        self.lengths_from = np.zeros(self.nrecvs, dtype=np.int32)
        self.procs_from = np.zeros_like(self.lengths_from)

        requests = [None for i in range(self.nrecvs)]

        for i in range(self.nrecvs):
            requests[i] = self.comm.Irecv(
                    self.lengths_from[i:i+1], source=mpi.ANY_SOURCE, tag=tag
                    )

        for i in range(self.nblocks):
            self.comm.Send(np.array(lengths_to[i]), dest=procs_to[i], tag=tag)

        for i in range(self.nrecvs):
            requests[i].Wait(status=status)
            self.procs_from[i] = status.Get_source()

        self.set_recv_info(self.procs_from, self.lengths_from)


class ObjectExchange(object):
    def __init__(self, backend=None):
        self.backend = get_backend(backend)
        self.import_proclist = None
        self.import_gids = None
        self.export_proclist = None
        self.export_gids = None

    def _find_import_lists(self):
        import_plan = Comm(self.export_proclist, backend=self.backend)
        if import_plan.nreturn:
            self.import_gids = carr.empty(import_plan.nreturn,
                                          backend=self.backend)
        import_plan.comm_do(self.export_gids, self.import_gids)

    def _find_export_lists(self):
        pass

    def invert_lists(self):
        pass
