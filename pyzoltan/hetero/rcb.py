import mpi4py.MPI as mpi
import numpy as np
import compyle.array as carr
from compyle.array import get_backend, wrap, update_minmax
from compyle.api import annotate
from compyle.parallel import Scan, Reduction, Elementwise
from compyle.template import Template
from pyzoltan.hetero.comm import Comm, CommBase, dtype_to_mpi, get_elwise, get_scan
from pytools import memoize


def dbg_print(msg):
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    print("Rank %s: %s" % (rank, msg))


def root_print(msg):
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        print(msg)


@annotate
def inp_partition_domain(i, weights):
    return weights[i]


@annotate
def out_partition_domain(i, item, prev_item, target_w, target_idx):
    if item >= target_w and prev_item < target_w:
        target_idx[0] = i


@annotate
def fill_new_proclist(i, new_proclist, displs, size):
    for j in range(size):
        if i >= displs[size - j - 1]:
            new_proclist[i] = size - j - 1
            return


class PointAssign(Template):
    def __init__(self, name, ndims):
        super(PointAssign, self).__init__(name=name)
        self.ndims = ndims
        self.args = []
        for i in range(self.ndims):
            self.args.append('x%s' % i)

    def extra_args(self):
        return self.args, {}

    def template(self, i, new_proc, mins, maxs, global_mins, global_maxs, size):
        '''
        % for dim in range(obj.ndims):
        gmax_${dim} = global_maxs[${dim}]
        gmin_${dim} = global_mins[${dim}]
        % endfor
        for j in range(size):
            count = 0

            % for dim in range(obj.ndims):
            pmax_dim = maxs[j * ${obj.ndims} + ${dim}]
            pmin_dim = mins[j * ${obj.ndims} + ${dim}]

            if x${dim}[i] <= pmax_dim and x${dim}[i] >= pmin_dim:
                count += 1
            elif x${dim}[i] <= pmax_dim and x${dim}[i] < gmin_${dim}:
                count += 1
            elif x${dim}[i] > gmax_${dim} and x${dim}[i] >= pmin_dim:
                count += 1
            % endfor

            if count == ${obj.ndims}:
                new_proc[i] = j
                return
        '''


class InpBoxAssignLength(Template):
    def __init__(self, name, ndims):
        super(InpBoxAssignLength, self).__init__(name=name)
        self.ndims = ndims
        self.args = []
        for i in range(self.ndims):
            self.args.append('x%s' % i)

    def extra_args(self):
        return self.args, {}

    def template(self, i, num_procs, self_proc, pmins, pmaxs, blength, size):
        '''
        proc_count = 0

        for j in range(size):
            if j == self_proc:
                continue

            count = 0

            % for dim in range(obj.ndims):
            pmax_dim = pmaxs[j * ${obj.ndims} + ${dim}]
            pmin_dim = pmins[j * ${obj.ndims} + ${dim}]

            plength_dim = pmax_dim - pmin_dim
            pcentre_dim = 0.5 * (pmax_dim + pmin_dim)

            if abs(x${dim}[i] - pcentre_dim) <= 0.5 * (plength_dim + blength):
                count += 1

            % endfor

            if count == ${obj.ndims}:
                proc_count += 1

        return proc_count
        '''


def out_box_assign_length(i, item, num_procs):
    num_procs[i] = item


class BoxAssign(Template):
    def __init__(self, name, ndims):
        super(BoxAssign, self).__init__(name=name)
        self.ndims = ndims
        self.args = []
        for i in range(self.ndims):
            self.args.append('x%s' % i)

    def extra_args(self):
        return self.args, {}

    def template(self, i, starts, nbr_procs, self_proc, pmins, pmaxs,
                 blength, size):
        '''
        proc_count = 0
        start = starts[i]

        for j in range(size):
            if j == self_proc:
                continue

            count = 0

            % for dim in range(obj.ndims):
            pmax_dim = pmaxs[j * ${obj.ndims} + ${dim}]
            pmin_dim = pmins[j * ${obj.ndims} + ${dim}]

            plength_dim = pmax_dim - pmin_dim
            pcentre_dim = 0.5 * (pmax_dim + pmin_dim)

            if abs(x${dim}[i] - pcentre_dim) <= 0.5 * (plength_dim + blength):
                count += 1

            % endfor

            if count == ${obj.ndims}:
                nbr_procs[start + proc_count] = j
                proc_count += 1
        '''


@memoize(key=lambda args: tuple(args))
def get_box_assign_length_kernel(ndims, backend):
    inp_box_assign_length = InpBoxAssignLength(
            'inp_box_assign_length', ndims
            )
    box_assign_length_knl = get_scan(
            inp_box_assign_length, out_box_assign_length,
            np.int32, backend
            )
    return box_assign_length_knl


@memoize(key=lambda args: tuple(args))
def get_box_assign_kernel(ndims, backend):
    box_assign = InpBoxAssignLength(
            'box_assign', ndims
            )
    box_assign_knl = get_elwise(box_assign, backend)
    return box_assign_knl


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
                 proc_weights=None, root=0, tag=111, padding=0.,
                 backend=None):
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
        self.plan_transfers = []
        self.padding = padding
        self.nobjs_per_proc = None
        self.all_gids = None
        self.displs = None
        self.lb_done = False
        self.adjusted = False

        self.point_assign_f = PointAssign('point_assign', self.ndims).function
        self.point_assign_knl = Elementwise(self.point_assign_f,
                                            backend=self.backend)

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
        elif self.rank == self.root:
            self.weights.dev /= carr.sum(self.weights)

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

        if self.rank == self.root:
            self.all_exec_times = np.zeros(self.size, dtype=np.float32)
            self.all_exec_nobjs = np.zeros(self.size, dtype=np.int32)
            self.num_objs = self.gids.length
        else:
            self.all_exec_times = None
            self.all_exec_nobjs = None
            self.num_objs = 0

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
        # FIXME: Clean up this mess

        order = carr.argsort(self.coords_view[maxlen_idx])

        align_coords = []
        for i, x in enumerate(self.coords_view):
            if i != maxlen_idx:
                align_coords.append(x)

        inp_list = [self.gids_view, self.weights_view] + align_coords

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
        self.weights_view = self.out_list[1].copy()

        curr_head = 2
        for i in range(self.ndims):
            if i != maxlen_idx:
                out = self.out_list[curr_head]
                self.coords_view[i] = out.copy()
                curr_head += 1

        target_idx = carr.zeros(1, dtype=np.int32, backend=self.backend)

        split_idx_knl = Scan(inp_partition_domain, out_partition_domain,
                             'a+b', dtype=self.weights_view.dtype,
                             backend=self.backend)

        split_idx_knl(weights=self.weights_view, target_w=target_w,
                      target_idx=target_idx)

        target_idx = int(target_idx[0])

        return maxlen_idx, target_idx

    def gather(self, data=[]):
        nobjs_to_get = None
        if self.rank == self.root:
            nobjs_to_get = np.zeros(self.size,
                                    dtype=self.gids_view.dtype)

        nobjs = self.gids_view.length

        self.comm.Gather(np.array(nobjs, dtype=np.int32),
                         nobjs_to_get, root=self.root)

        if self.rank == self.root:
            total_nobjs = np.sum(nobjs_to_get)
            gath_gids = carr.empty(total_nobjs, np.int32,
                                   backend=self.backend)
            gath_weights = carr.empty(total_nobjs, np.float32,
                                      backend=self.backend)
            gath_coords = [carr.empty(total_nobjs, self.dtype, backend=self.backend) \
                    for i in range(self.ndims)]
            gath_proc_weights = carr.empty(self.size, np.float32,
                                           backend=self.backend)
            gath_data = [carr.empty(total_nobjs, x.dtype, backend=self.backend) \
                    for x in data]

            gath_gids_buff = gath_gids.get_buff()
            gath_weights_buff = gath_weights.get_buff()
            gath_coords_buff = [x.get_buff() for x in gath_coords]
            gath_proc_weights_buff = gath_proc_weights.get_buff()
            gath_data_buff = [x.get_buff() for x in gath_data]

            displs = np.zeros(self.size, dtype=np.int32)
            np.cumsum(nobjs_to_get[:-1], out=displs[1:])
        else:
            gath_gids, gath_weights, gath_proc_weights = None, None, None
            gath_coords = [None] * self.ndims
            gath_data = [None] * len(data)

            gath_gids_buff = None
            gath_weights_buff = None
            gath_coords_buff = [None] * self.ndims
            gath_proc_weights_buff = None
            gath_data_buff = [None] * len(data)

            displs = None

        self.comm.Gatherv(sendbuf=[self.gids_view.get_buff(),
                                   dtype_to_mpi(self.gids_view.dtype)],
                          recvbuf=[gath_gids_buff, nobjs_to_get, displs,
                                   dtype_to_mpi(self.gids_view.dtype)],
                          root=self.root)

        self.comm.Gatherv(sendbuf=[self.weights_view.get_buff(),
                                   dtype_to_mpi(self.weights_view.dtype)],
                          recvbuf=[gath_weights_buff, nobjs_to_get, displs,
                                   dtype_to_mpi(self.weights_view.dtype)],
                          root=self.root)

        for i, x in enumerate(self.coords_view):
            self.comm.Gatherv(sendbuf=[x.get_buff(),
                                       dtype_to_mpi(x.dtype)],
                              recvbuf=[gath_coords_buff[i], nobjs_to_get, displs,
                                       dtype_to_mpi(x.dtype)],
                              root=self.root)

        if not self.adjusted:
            self.comm.Gather([self.proc_weights_view.get_buff(),
                              dtype_to_mpi(self.proc_weights_view.dtype)],
                              gath_proc_weights_buff,
                              root=self.root)
        elif self.rank == self.root:
            # proc weights are reset during adjustment
            gath_proc_weights = self.proc_weights_view

        for i, ary in enumerate(data):
            self.comm.Gatherv(sendbuf=[ary.get_buff(),
                                       dtype_to_mpi(ary.dtype)],
                              recvbuf=[gath_data_buff[i], nobjs_to_get, displs,
                                       dtype_to_mpi(ary.dtype)],
                              root=self.root)

        self.gids_view = gath_gids
        self.weights_view = gath_weights
        self.proc_weights_view = gath_proc_weights
        if self.rank == self.root:
            self.coords_view = gath_coords
        else:
            self.coords_view = []

        return gath_data

    def adjust_proc_weights(self, exec_time, exec_nobjs, exec_count):
        # set new proc weights based on exec times
        # send all times to root
        scale_fac = self.total_num_objs * (exec_time / exec_nobjs)

        self.comm.Gather(np.array(exec_time, dtype=np.float32),
                            self.all_exec_times, root=self.root)

        self.comm.Gather(np.array(exec_nobjs, dtype=np.int32),
                            self.all_exec_nobjs, root=self.root)

        if self.rank == self.root:
            eff_weights = self.all_exec_nobjs / \
                    (self.total_num_objs * exec_count)
            scale_fac = 1. / np.sum(eff_weights / self.all_exec_times)
            proc_weights = scale_fac * eff_weights / self.all_exec_times
            proc_weights = proc_weights.astype(np.float32)

            self.proc_weights = wrap(proc_weights, backend=self.backend)
            self.proc_weights_view = self.proc_weights.get_view()

            self.all_exec_times = np.zeros(self.size, dtype=np.float32)
            self.all_exec_nobjs = np.zeros(self.size, dtype=np.int32)

        self.adjusted = True

    def make_migrate_plan(self):
        new_proc = carr.empty(self.num_objs, np.int32, backend=self.backend)
        new_proc.fill(self.rank)
        self.point_assign_knl(new_proc, self.mins, self.maxs,
                              self.global_mins, self.global_maxs,
                              self.size, *self.coords_view)
        self.migrate_plan = Comm(new_proc, root=self.root, backend=self.backend)
        return new_proc

    def migrate_objects(self, data):
        new_proc = self.make_migrate_plan()

        new_len = self.migrate_plan.nreturn
        nsends = self.migrate_plan.nblocks
        # gids, weights, coords, data

        if nsends == 1 and new_len == self.num_objs and new_proc[0] == self.rank:
            return

        new_coords = []
        new_data = []

        new_gids = carr.empty(new_len, self.gids_view.dtype,
                              backend=self.backend)
        new_weights = carr.empty(new_len, self.weights_view.dtype,
                                 backend=self.backend)
        for i, x in enumerate(self.coords_view):
            new_coords.append(carr.empty(new_len, x.dtype,
                                         backend=self.backend))

        for i, x in enumerate(data):
            new_data.append(carr.empty(new_len, x.dtype, backend=self.backend))

        self.migrate_plan.comm_do_post(self.gids_view, new_gids)
        self.migrate_plan.comm_do_post(self.weights_view, new_weights)

        for i, x in enumerate(self.coords_view):
            self.migrate_plan.comm_do_post(x, new_coords[i])

        for i, x in enumerate(data):
            self.migrate_plan.comm_do_post(x, new_data[i])

        self.migrate_plan.comm_do_wait()

        self.gids_view.set_data(new_gids)
        self.weights_view.set_data(new_weights)
        for i, x_new in enumerate(new_coords):
            self.coords_view[i].set_data(x_new)
        for i, x_new in enumerate(new_data):
            data[i].set_data(x_new)

        self.num_objs = int(new_len)

    def update_bounds(self):
        if self.rank == self.root:
            if self.coords_view:
                update_minmax(self.coords_view)

            for i, x in enumerate(self.coords_view):
                xlength = x.maximum - x.minimum
                eps = 0.
                if xlength == 0:
                    eps = 10 * np.finfo(np.float32).eps
                self.max[i] = eps + x.maximum + self.padding * xlength
                self.min[i] = -eps + x.minimum - self.padding * xlength

    def load_balance_raw(self):
        # FIXME: These tags might mess with other sends and receives
        reqs_coords = []
        req_objids = None
        req_w = None
        req_max, req_min = None, None

        if self.rank == self.root:
            self.procs_view = carr.arange(0, self.size, 1, dtype=np.int32,
                                          backend=self.backend)
        else:
            self.procs_view = None

        self.update_bounds()

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
                req_max.Wait()
                req_min.Wait()
                req_w.Wait()
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
            part_pos = 0.5 * (part_dim[target_idx - 1] + part_dim[target_idx])

            self.max[maxlen_idx] = part_pos
            right_min[maxlen_idx] = part_pos

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

    def load_balance(self):
        if self.lb_done:
            self.gather()
        self.old_all_gids = self.gids_view
        self.load_balance_raw()
        self.make_comm_plan()

    def make_comm_plan(self):
        # Send object ids back to root
        old_num_objs = self.num_objs
        # FIXME: Need to gather all gids from the previous iteration
        # it is done anyways!
        old_nobjs_per_proc = self.nobjs_per_proc
        old_displs = self.displs

        self.num_objs = self.gids_view.length
        self.nobjs_per_proc = None

        if self.rank == self.root:
            self.nobjs_per_proc = np.zeros(self.size, dtype=self.gids_view.dtype)

        self.comm.Gather(np.array(self.num_objs, dtype=np.int32),
                         self.nobjs_per_proc, root=self.root)

        root_print("Current nobjs/proc = %s, num_objs = %s" % (self.nobjs_per_proc, self.num_objs))

        if self.rank == self.root:
            self.total_num_objs = np.sum(self.nobjs_per_proc,
                                         dtype=self.gids_view.dtype)
            self.all_gids = carr.empty(self.total_num_objs, np.int32,
                                       backend=self.backend)
            all_gids_buff = self.all_gids.get_buff()
            self.displs = np.zeros(self.size, dtype=np.int32)
            np.cumsum(self.nobjs_per_proc[:-1], out=self.displs[1:])
        else:
            self.total_num_objs = np.array(0, dtype=self.gids_view.dtype)
            self.all_gids = None
            all_gids_buff = None
            self.displs = None

        self.comm.Bcast(self.total_num_objs, root=self.root)

        self.maxs = carr.empty(self.ndims * self.size, self.dtype,
                               backend=self.backend)
        self.mins = carr.empty(self.ndims * self.size, self.dtype,
                               backend=self.backend)

        self.comm.Allgather(self.max, self.maxs.get_buff())
        self.comm.Allgather(self.min, self.mins.get_buff())

        np_maxs = self.maxs.get()
        np_mins = self.mins.get()

        global_maxs = np.empty(self.ndims, dtype=self.maxs.dtype)
        global_mins = np.empty(self.ndims, dtype=self.mins.dtype)

        # FIXME: This can be optimized if required
        for i in range(self.ndims):
            global_maxs[i] = np_maxs[i::self.ndims].max()
            global_mins[i] = np_mins[i::self.ndims].min()

        self.global_maxs = wrap(global_maxs, backend=self.backend)
        self.global_mins = wrap(global_mins, backend=self.backend)

        self.comm.Gatherv(sendbuf=[self.gids_view.get_buff(),
                                   dtype_to_mpi(self.gids_view.dtype)],
                          recvbuf=[all_gids_buff, self.nobjs_per_proc,
                                   self.displs,
                                   dtype_to_mpi(self.gids_view.dtype)],
                          root=self.root)

        if not self.lb_done:
            # FIXME: tag
            self.plan = CommBase(None, sorted=True, root=self.root, tag=self.tag)
            procs_from = np.array([0], dtype=np.int32)
            lengths_from = np.array([self.num_objs], dtype=np.int32)

            if self.rank == self.root:
                procs_to = np.arange(0, self.size, dtype=np.int32)
                lengths_to = self.nobjs_per_proc
            else:
                procs_to = np.array([], dtype=np.int32)
                lengths_to = np.array([], dtype=np.int32)

            self.plan.set_send_info(procs_to, lengths_to)
            self.plan.set_recv_info(procs_from, lengths_from)
        else:
            # find the new proc for each object
            # then create a plan with the new procs using the old gids

            # make a new procs array and align it according to new all_gids
            self.comm.Gather(np.array(old_num_objs, dtype=np.int32),
                             old_nobjs_per_proc, root=self.root)

            if self.rank == self.root:
                np.cumsum(old_nobjs_per_proc[:-1], out=old_displs[1:])

                new_proclist = carr.empty(self.total_num_objs, np.int32,
                                          backend=self.backend)

                fill_new_proclist_knl = get_elwise(fill_new_proclist,
                                                   backend=self.backend)

                # FIXME: Avoid wrapping this
                displs_arr = wrap(self.displs, backend=self.backend)
                fill_new_proclist_knl(new_proclist, displs_arr, self.size)

                semi_aligned_proclist = new_proclist.align(self.all_gids)

                semi_aligned_proclist.align(self.old_all_gids, out=new_proclist)

                new_proclist_buff = new_proclist.get_buff()

                dbg_print(new_proclist)
            else:
                new_proclist_buff = None

            self.transfer_proclist = carr.empty(old_num_objs,
                                                np.int32, backend=self.backend)
            #dbg_print("%s %s %s %s" % (old_displs, old_nobjs_per_proc, old_num_objs, self.num_objs))
            dbg_print("num_objs= %s" % old_num_objs)
            root_print("Num objs = %s, displs = %s" % (old_nobjs_per_proc, old_displs))

            self.comm.Scatterv(sendbuf=[new_proclist_buff,
                                        old_nobjs_per_proc, old_displs,
                                        dtype_to_mpi(np.int32)],
                               recvbuf=[self.transfer_proclist.get_buff(),
                                        dtype_to_mpi(np.int32)],
                               root=self.root)

            #dbg_print(self.transfer_proclist)

            self.plan = Comm(self.transfer_proclist, backend=self.backend)

        self.lb_done = True

    def find_nbr_procs(self, box_size):
        box_assign_length_knl = get_box_assign_length_kernel(
                self.ndims, self.backend
                )

        num_procs = carr.zeros(self.num_objs, np.int32, backend=self.backend)

        coord_args = {}
        for i, x in enumerate(self.coords_view):
            coord_args['x%s' % i] = x

        box_assign_length_knl(num_procs=num_procs, self_proc=self.rank,
                              pmins=self.mins, pmaxs=self.maxs,
                              blength=box_size, size=self.size, **coord_args)

        starts = carr.zeros_like(num_procs)

        carr.cumsum(num_procs[:-1], out=starts[1:])

        num_nbr_procs = num_procs[-1] + starts[-1]

        nbr_procs = carr.empty(num_nbr_procs, backend=self.backend)

        box_assign_knl = get_box_assign_kernel(self.ndims, self.backend)

        box_assign_knl(starts, nbr_procs, self.rank, self.mins, self.maxs,
                       box_size, self.size, *self.coords_view)

    def comm_do_post(self, senddata, recvdata):
        self.plan.comm_do_post(senddata, recvdata)

    def comm_do_wait(self):
        self.plan.comm_do_wait()

    def comm_do(self, senddata, recvdata):
        self.plan.comm_do(senddata, recvdata)


@annotate
def flatten1(x, ncells_per_dim):
    res = declare('long')
    return res


@annotate
def flatten3(x, y, z, ncells_per_dim):
    ncx = ncells_per_dim[0]
    ncy = ncells_per_dim[1]
    res = declare('long')
    res = x + ncx * y + ncx * ncy * z
    return res


@annotate
def unflatten1(key, cid, ncells_per_dim):
    cid[0] = cast(key, 'int')


@annotate
def unflatten2(key, cid, ncells_per_dim):
    ncx = ncells_per_dim[0]
    cid[1] = cast(key / ncx, 'int')
    cid[0] = cast(key - cid[1] * ncx, 'int')


@annotate
def unflatten3(key, cid, ncells_per_dim):
    ncx = ncells_per_dim[0]
    ncy = ncells_per_dim[1]
    cid[2] = cast(key / (ncx * ncy), 'int')
    cid[1] = cast((key - cid[2] * ncx * ncy) / ncx, 'int')
    cid[0] = cast(key - cid[1] * ncx - cid[2] * ncy * ncx, 'int')


@annotate
def flatten2(x, y, ncells_per_dim):
    ncx = ncells_per_dim[0]
    res = declare('long')
    res = x + ncx * y
    return res


class FillKeys(Template):
    def __init__(self, name, ndims):
        super(FillKeys, self).__init__(name=name)
        self.ndims = ndims
        self.args = []
        for i in range(self.ndims):
            self.args.append('x%s' % i)
        self.int_args = ['c%s' % j for j in range(self.ndims)]
        self.int_args = ', '.join(self.int_args)

    def extra_args(self):
        min_args = ['%smin' % arg for arg in self.args]
        return self.args + min_args, {}

    def template(self, i, keys, bin_size, ncells_per_dim):
        '''
        ncells_per_dim = declare('matrix(${obj.ndims}), int')
        % for j, x in enumerate(obj.args)
        c${j} = floor((${x}[i] - ${x}min) / bin_size)
        % endfor
        keys[i] = flatten${obj.ndims}(${obj.int_args}, ncells_per_dim)
        '''


@memoize(key=lambda *args: tuple(args))
def get_fill_keys_kernel(ndims, backend):
    fill_keys = FillKeys('fill_keys', ndims).function
    fill_keys_knl = get_elwise(fill_keys, backend)
    return fill_keys_knl


@annotate
def inp_fill_centroids(i, keys):
    return 1 if i != 0 and keys[i] != keys[i - 1] else 0


class OutFillCentroids(Template):
    def __init__(self, name, ndims):
        super(OutFillCentroids, self).__init__(name=name)
        self.ndims = ndims
        self.args = []
        for i in range(self.ndims):
            self.args.append('centroid%s' % i)

    def extra_args(self):
        coord_names = ['x%s' % i for i in range(self.ndims)]
        min_args = ['%smin' % arg for arg in coord_names]
        return self.args, {}

    def template(self, i, item, prev_item, ncells_per_dim,
                 bin_size, cell_to_key, cell_to_idx, key_to_idx):
        '''
        if i == 0 or item != prev_item:
            key = keys[i]
            cids = declare('matrix(${obj.ndims}), int')
            unflatten${obj.ndims}(keys[i], cids, ncells_per_dim)
            % for dim in range(obj.ndims)
            centroid${dim}[item] = x${dim}min + bin_size * (0.5 + cids[${dim}])
            % endfor
            cell_to_key[item] = key
            cell_to_idx[item] = i
            key_to_idx[key] = i
        '''


@annotate
def fill_cell_weights(i, cell_to_idx, cell_weights, cell_num_objs, weights,
                      num_cells):
    start = cell_to_idx[i]
    if i < num_cells - 1:
        end = cell_to_idx[i + 1]
    else:
        end = num_cells
    num_objs = end - start

    tot_weight = 0.
    for j in range(num_objs):
        tot_weight += weights[start + j]

    cell_num_objs[i] = num_objs
    cell_weights[i] = tot_weight


@annotate
def inp_fill_gids_from_cids(i, cell_num_objs):
    cid = all_cids[i]
    return cell_num_objs[cid]


@annotate
def out_fill_gids_from_cids(i, all_cids, all_gids, cell_to_idx,
                            cell_num_objs, gids):
    cid = all_cids[i]
    start_idx = cell_to_idx[cid]
    nobjs = cell_num_objs[cid]

    for j in range(nobjs):
        all_gids[prev_item + j] = gids[start_idx + j]


@memoize(key=lambda *args: tuple(args))
def get_fill_centroids_kernel(ndims, backend):
    out_fill_centroids = OutFillCentroids(
            'out_fill_centroids', ndims
            ).function
    fill_centroids_knl = get_scan(inp_fill_centroids, out_fill_centroids,
                                  np.uint64, backend)
    return fill_centroids_knl



class BinnedRCB(object):
    def __init__(self, ndims, dtype, coords=[], gids=None, weights=None,
                 proc_weights=None, bin_size=None, root=0, tag=111,
                 backend=None):
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
        elif self.rank == self.root:
            self.weights.dev /= carr.sum(self.weights)

        if self.rank == self.root and self.proc_weights is None:
            self.proc_weights = carr.empty(self.size, np.float32,
                                           backend=self.backend)
            self.proc_weights.fill(1. / self.size)

        # views of original arrays
        self.procs_view = self.procs.get_view() if self.procs else None
        self.proc_weights_view = self.proc_weights.get_view() \
                if self.proc_weights else None
        self.weights_view = self.weights.get_view() if self.weights else None
        self.gids_view = self.gids.get_view() if self.gids else None
        self.coords_view = [x.get_view() for x in self.coords] \
                if self.coords else []

        self.max = np.zeros(self.ndims, dtype=self.dtype)
        self.min = np.zeros(self.ndims, dtype=self.dtype)

        if self.coords_view:
            update_minmax(self.coords_view)

        for i, x in enumerate(self.coords_view):
            self.max[i] = x.maximum
            self.min[i] = x.minimum

        self.centroids = []
        self.cids = None
        self.cell_weights = None

        self.bin_size = bin_size

        if self.rank == self.root:
            self.bin()

        self.rcb = RCB(self.ndims, self.dtype, coords=self.centroids,
                       gids=self.cids, weights=self.cell_weights,
                       proc_weights=self.proc_weights, root=self.root,
                       tag=self.tag, backend=self.backend)

    def bin(self):
        fill_keys_knl = get_fill_keys_knl(self.ndims, self.backend)
        fill_centroids_knl = get_fill_centroids_kernel(ndims, self.backend)
        fill_cell_weights_knl = get_elwise('fill_cell_weights',
                                           backend=self.backend)

        self.keys = carr.empty(self.gids_view.length, np.int64,
                               backend=self.backend)

        fill_keys_args = [self.keys] + self.coords_view + list(self.min)

        fill_keys_knl(*fill_keys_args)

        # sort
        inp_list = [self.keys, self.gids_view] + self.coords_view
        out_list = carr.sort_by_keys(inp_list, backend=self.backend)
        self.keys, self.gids_view = out_list[0], out_list[1]
        self.coords_view = out_list[2:]

        self.key_to_idx = carr.zeros(1 + int(self.keys[-1]), np.int32,
                                     backend=self.backend)

        num_cids_knl = Reduction('a+b', map_func=inp_fill_centroids,
                                 dtype_out=np.int32, backend=self.backend)

        num_cids = int(num_cids_knl(self.keys))

        self.cell_to_key = carr.empty(num_cids, np.int64,
                                      backend=self.backend)
        self.cell_to_idx = carr.empty(num_cids, np.int32,
                                       backend=self.backend)
        self.cell_weights = carr.empty(num_cids, np.float32,
                                       backend=self.backend)
        for dim in range(self.ndims):
            self.centroids.append(carr.empty(num_cids, self.dtype,
                                             backend=self.backend))

        ncells_per_dim = np.empty(self.ndims, dtype=np.int32)

        for i in range(self.ndims):
            ncells_per_dim[i] = \
                    np.ceil((self.max[i] - self.min[i]) / self.bin_size)

        ncells_per_dim = wrap(ncells_per_dim, backend=self.backend)

        fill_centroids_args = {'keys': self.keys,
                'ncells_per_dim': ncells_per_dim,
                'bin_size': bin_size, 'cell_to_key': self.cell_to_key,
                'key_to_idx': self.key_to_idx,
                'cell_to_idx': self.cell_to_idx}

        for dim in range(self.ndims):
            arg_name = 'x%s' % dim
            min_arg_name = 'x%smin' % dim
            cen_arg_name = 'centroid%s' % dim
            fill_centroids_args[arg_name] = self.coords_view[dim]
            fill_centroids_args[min_arg_name] = self.min[dim]
            fill_centroids_args[cen_arg_name] = self.centroids[dim]

        fill_centroids_knl(**fill_centroids_args)

        self.cids = carr.arange(0, num_cids, 1, np.int32,
                                backend=self.backend)

        fill_cell_weights_knl(self.cell_to_idx, self.cell_weights,
                              self.cell_num_objs, self.weights,
                              self.cell_to_idx.length)

    def load_balance_raw(self):
        self.rcb.load_balance_raw()

    def load_balance(self):
        self.load_balance_raw()
        self.make_comm_plan()

    def make_comm_plan(self):
        # Send object ids back to root
        self.ncells_per_proc = None
        if self.rank == self.root:
            self.ncells_per_proc = np.zeros(self.size,
                                            dtype=self.rcb.gids_view.dtype)

        self.comm.Gather(np.array(self.rcb.num_objs, dtype=np.int32),
                         self.ncells_per_proc, root=self.root)

        if self.rank == self.root:
            self.total_num_cells = np.sum(self.ncells_per_proc,
                                          dtype=self.rcb.gids_view.dtype)
            self.all_cids = carr.empty(self.total_num_cells, np.int32,
                                       backend=self.backend)
            all_cids_buff = self.all_cids.get_buff()
            self.all_gids = carr.empty(total_num_objs, np.int32,
                                       backend=self.backend)
            displs = np.zeros(self.size, dtype=np.int32)
            np.cumsum(self.ncells_per_proc[:-1], out=displs[1:])
        else:
            self.total_num_cells = np.array(0, dtype=self.gids_view.dtype)
            self.all_cids = None
            all_cids_buff = None
            displs = None

        self.comm.Bcast(self.total_num_cells, root=self.root)

        self.maxs = carr.empty(self.ndims * self.size, self.dtype,
                               backend=self.backend)
        self.mins = carr.empty(self.ndims * self.size, self.dtype,
                               backend=self.backend)

        self.comm.Allgather(self.max, self.maxs.get_buff())
        self.comm.Allgather(self.min, self.mins.get_buff())

        self.comm.Gatherv(sendbuf=[self.rcb.gids_view.get_buff(),
                                   dtype_to_mpi(self.rcb.gids_view.dtype)],
                          recvbuf=[all_cids_buff, self.ncells_per_proc, displs,
                                   dtype_to_mpi(self.rcb.gids_view.dtype)],
                          root=self.root)

        if self.rank == self.root:
            fill_gids_from_cids_knl = get_scan(inp_fill_gids_from_cids,
                                               out_fill_gids_from_cids,
                                               np.int32, self.backend)
            fill_gids_from_cids_knl(all_cids=all_cids, all_gids=all_gids,
                                    cell_to_idx=self.cell_to_idx,
                                    cell_num_objs=self.cell_num_objs)

        # FIXME: tag
        self.plan = CommBase(None, sorted=True, root=self.root, tag=self.tag)
        procs_from = np.array([0], dtype=np.int32)
        lengths_from = np.array([self.rcb.num_objs], dtype=np.int32)

        if self.rank == self.root:
            procs_to = np.arange(0, self.size, dtype=np.int32)
            lengths_to = self.ncells_per_proc
        else:
            procs_to = np.array([], dtype=np.int32)
            lengths_to = np.array([], dtype=np.int32)

        self.plan.set_send_info(procs_to, lengths_to)
        self.plan.set_recv_info(procs_from, lengths_from)


