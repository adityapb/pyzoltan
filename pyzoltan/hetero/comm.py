import mpi4py.MPI as mpi
import numpy as np
import compyle.array as carr
from compyle.array import get_backend
from compyle.types import NP_TYPE_LIST


def dtype_to_mpi(t):
    if hasattr(mpi, '_typedict'):
        mpi_type = mpi._typedict[np.dtype(t).char]
    elif hasattr(mpi, '__TypeDict__'):
        mpi_type = mpi.__TypeDict__[np.dtype(t).char]
    else:
        raise ValueError('cannot convert type')
    return mpi_type


class Comm(object):
    def __init__(self, proclist, tag=0, root=0, backend=None, sorted=False):
        self.comm = mpi.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.sorted = sorted
        self.backend = get_backend(backend)
        self.root = root
        self.nsends = len(proclist)
        self.proclist = proclist
        self.tag = tag
        if len(self.proclist) > 0:
            if not self.sorted:
                self.order = carr.wrap_array(self._sort_proclist(),
                                             backend=self.backend)
            starts, procs_to, lengths_to = [0], [proclist[0]], [1]
            for i in range(1, self.nsends):
                if proclist[i] != proclist[i-1]:
                    starts.append(i)
                    procs_to.append(proclist[i])
                    lengths_to.append(1)
                else:
                    lengths_to[-1] += 1
            self.starts = np.array(starts, dtype=np.int32)
            self.procs_to = np.array(procs_to, dtype=np.int32)
            self.lengths_to = np.array(lengths_to, dtype=np.int32)
            self.nblocks = self.procs_to.size
        else:
            self.starts = np.empty(0, dtype=np.int32)
            self.procs_to = np.empty(0, dtype=np.int32)
            self.lengths_to = np.empty(0, dtype=np.int32)
            self.nblocks = 0
        self._comm_invert_map(tag, self.lengths_to, self.procs_to)
        self.blocked_senddata = None

    def _sort_proclist(self):
        order = np.argsort(self.proclist)
        self.proclist = self.proclist[order]
        return order

    def _comm_invert_map(self, tag, lengths_to, procs_to):
        status = mpi.Status()
        msg_count = np.zeros(self.size, dtype=np.int32)
        counts = np.ones_like(msg_count)
        nrecvs = np.zeros(1, dtype=np.int32)

        for proc in procs_to:
            msg_count[proc] = 1

        self.comm.Reduce(msg_count, counts, root=self.root)
        self.comm.Scatter(counts, nrecvs, root=self.root)
        self.nrecvs = int(nrecvs)

        self.lengths_from = np.zeros(self.nrecvs, dtype=np.int32)
        self.procs_from = np.zeros_like(self.lengths_from)

        self.requests = [None for i in range(self.nrecvs)]

        for i in range(self.nrecvs):
            self.requests[i] = self.comm.Irecv(
                    self.lengths_from[i:i+1], source=mpi.ANY_SOURCE, tag=tag
                    )

        for i in range(self.nblocks):
            self.comm.Send(np.array(lengths_to[i]), dest=procs_to[i], tag=tag)

        for i in range(self.nrecvs):
            self.requests[i].Wait(status=status)
            self.procs_from[i] = status.Get_source()

        ord_procs_from = np.argsort(self.procs_from)
        self.procs_from = self.procs_from[ord_procs_from]
        self.lengths_from = self.lengths_from[ord_procs_from]
        self.start_from = np.zeros_like(self.lengths_from)
        np.cumsum(self.lengths_from[:-1], out=self.start_from[1:])
        self.nreturn = np.sum(self.lengths_from)

    def get_recvdtype(self, senddtype):
        recvdtype = np.zeros_like(self.procs_from)
        senddtype = NP_TYPE_LIST.index(senddtype)

        if senddtype == -1:
            raise ValueError("Invalid type for send array")

        for i in range(self.nrecvs):
            self.requests[i] = self.comm.Irecv(
                    recvdtype[i:i+1], source=self.procs_from[i],
                    tag=self.tag
                    )

        for i in range(self.nblocks):
            self.comm.Send(np.array(senddtype, dtype=np.int32),
                           dest=self.procs_to[i], tag=self.tag)

        mpi.Request.Waitall(self.requests)

        if not np.all(recvdtype == recvdtype[0]):
            raise ValueError("All sends should have same datatype")

        return NP_TYPE_LIST[recvdtype[0]]

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

        self.requests = []

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

    def comm_do(self, senddata, recvdata):
        self.comm_do_post(senddata, recvdata)
        self.comm_do_wait()
