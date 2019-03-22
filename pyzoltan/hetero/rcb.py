import mpi4py.MPI as mpi
import numpy as np
import compyle.array as carr


@annotate
def inp_partition_domain(i, weights):
    return weights[i]


@annotate
def out_partition_domain(i, item, prev_item, target_w, target_idx):
    if item > target_w and prev_item < target_w:
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
    def __init__(self, data, backend=None):
        self.comm = mpi.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.backend = get_backend(backend)
        self.data = data
        self.ndims = len(data)

    def _partition_procs(self):
        # NOTE: This can be made better using the weights
        # of the procs and communication
        part_idx = self.nprocs // 2
        target_w = carr.sum(self.proc_weights[:part_idx])
        return part_idx, target_w

    def _partition_domain(self, target_w):
        self.x.update_min_max()
        self.y.update_min_max()
        self.z.update_min_max()

        xlength = self.x.maximum - self.x.minimum
        ylength = self.y.maximum - self.y.minimum
        zlength = self.z.maximum - self.z.minimum

        if xlength > ylength and xlength > zlength:
            # sort according to x
            pass
        if ylength > xlength and ylength > zlength:
            # sort according to y
            pass
        if zlength > xlength and zlength > ylength:
            # sort according to z
            pass

        target_idx = carr.zeros(1, dtype=np.int32, backend=self.backend)

        split_idx_knl = Scan(inp_partition_domain, out_partition_domain,
                             'a+b', dtype=self.weights.dtype,
                             backend=self.backend)

        split_idx_knl(weights=self.weights, target_w=target_w,
                      target_idx=target_idx)

        return int(target_idx[0])

    def create_partitions(self, tag):
        # FIXME: fix tags for messages
        requests = []
        if self.rank != self.root:
            status = mpi.Status()
            # Wait to receive length of data
            nrecv_data = np.empty(1, dtype=np.int32)
            nrecv_procs = np.empty(1, dtype=np.int32)
            req_data = self.comm.Irecv(
                    nrecv_data, source=mpi.ANY_SOURCE, tag=tag
                    )

            req_procs = self.comm.Irecv(
                    nrecv_procs, source=mpi.ANY_SOURCE, tag=tag
                    )

            req_procs.Wait()
            self.nrecv_procs = int(nrecv_procs)
            self.procs = carr.empty(self.nrecv_procs, dtype=np.int32,
                                    backend=self.backend)

            req_data.Wait(status=status)
            self.parent = status.Get_source()

            self.nrecv_data = int(nrecv_data)

            self.x = carr.empty(self.nrecv_data, dtype=self.dtype,
                                backend=self.backend)
            self.y = carr.empty(self.nrecv_data, dtype=self.dtype,
                                backend=self.backend)
            self.z = carr.empty(self.nrecv_data, dtype=self.dtype,
                                backend=self.backend)

            req_procs = self.comm.Irecv(self.procs, source=self.parent,
                                        tag=tag)

            requests.append(self.comm.Irecv(self.x.get_buff(),
                            source=self.parent, tag=tag))
            requests.append(self.comm.Irecv(self.y.get_buff(),
                            source=self.parent, tag=tag))
            requests.append(self.comm.Irecv(self.z.get_buff(),
                            source=self.parent, tag=tag))

            req_procs.Wait()

            if len(self.procs) == 1:
                mpi.Request.Waitall(requests)
                return

        part_idx, target_w = self._partition_procs()

        if requests:
            mpi.Request.Waitall(requests)

        target_idx = self._partition_domain(target_w)

        left_proc = int(self.procs[0])
        right_proc = int(self.procs[part_idx])

        self.comm.Barrier()

        # transfer procs
        # left
        self.comm.mpi_Rsend(self.procs.get_buff(offset=0),
                            dest=left_proc, tag=tag)
        # right
        self.comm.mpi_Rsend(self.procs.get_buff(offset=part_idx),
                            dest=left_proc, tag=tag)

        # transfer x, y, z
        # left
        self.comm.mpi_Rsend(self.x.get_buff(offset=0),
                            dest=left_proc, tag=tag)
        self.comm.mpi_Rsend(self.y.get_buff(offset=0),
                            dest=left_proc, tag=tag)
        self.comm.mpi_Rsend(self.z.get_buff(offset=0),
                            dest=left_proc, tag=tag)
        # right
        self.comm.mpi_Rsend(self.x.get_buff(offset=target_idx),
                            dest=right_proc, tag=tag)
        self.comm.mpi_Rsend(self.y.get_buff(offset=target_idx),
                            dest=right_proc, tag=tag)
        self.comm.mpi_Rsend(self.z.get_buff(offset=target_idx),
                            dest=right_proc, tag=tag)


