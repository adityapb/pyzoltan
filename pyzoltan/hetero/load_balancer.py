import mpi4py.MPI as mpi
import numpy as np
import compyle.array as carr
from compyle.array import get_backend, wrap, update_minmax
from compyle.api import annotate
from compyle.parallel import Scan, Reduction, Elementwise
from compyle.template import Template
from compyle.low_level import cast
from pyzoltan.hetero.comm import (Comm, CommBase, dtype_to_mpi,
                                  get_elwise, get_scan, get_reduction)
from pyzoltan.hetero.cell_manager import (flatten1, flatten2, flatten3,
                                          unflatten1, unflatten2, unflatten3)
from pyzoltan.hetero.comm import dbg_print
from pytools import memoize


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


@annotate
def inp_fill_gids_from_cids(i, all_cids, cell_num_objs):
    cid = declare('int')
    cid = all_cids[i]
    return cell_num_objs[cid]


@annotate
def out_fill_gids_from_cids(i, item, prev_item, all_cids, all_gids, cell_to_idx,
                            cell_num_objs, gids):
    cid, start_idx, nobjs, j = declare('int', 4)
    cid = all_cids[i]
    start_idx = cell_to_idx[cid]
    nobjs = cell_num_objs[cid]

    for j in range(nobjs):
        all_gids[prev_item + j] = gids[start_idx + j]


class PointAssign(Template):
    def __init__(self, name, ndims):
        super(PointAssign, self).__init__(name=name)
        self.ndims = ndims
        self.args = []
        self.min_args = []
        for i in range(self.ndims):
            self.args.append('x%s' % i)
        for i in range(self.ndims):
            self.min_args.append('x%smin' % i)

    def extra_args(self):
        return self.args + self.min_args, {}

    def template(self, i, cids, new_proc, cell_size, cell_to_proc, key_to_cell,
                 max_key, ncells_per_dim):
        '''
        point = declare('matrix(${obj.ndims}, "int")')
        key = declare('long')
        % for j, x in enumerate(obj.args):
        point[${j}] = cast(floor((${x}[i] - ${x}min) / cell_size), 'int')
        % endfor
        key = flatten${obj.ndims}(point, ncells_per_dim)
        if key > max_key or key < 0:
            cid = -1
        else:
            cid = key_to_cell[key]
        if cid != -1:
            new_proc[i] = cell_to_proc[cid]
        '''


class InpBoxAssignLength(Template):
    def __init__(self, name, ndims):
        super(InpBoxAssignLength, self).__init__(name=name)
        self.ndims = ndims
        self.args = []
        for i in range(self.ndims):
            self.args.append('x%s' % i)
        self.indent = '    ' * self.ndims

    def extra_args(self):
        return self.args, {}

    def template(self, i, cids, cell_to_key, ncells_per_dim, key_to_cell,
                 cell_to_proc, self_proc):
        # Iterate over all objects and not all cells
        # Because of migrate particles, some objects may
        # be present in a non existent cell.
        '''
        proc_count = 0
        cid = cids[i]
        key = cell_to_key[cid]
        cell_coord = declare('matrix(${obj.ndims}, "int")')
        unflatten${obj.ndims}(key, cell_coord, ncells_per_dim):

        for p in range(-1, 2):
            cell_coord[0] += p
            % if obj.ndims == 1:
            nbr_key = flatten1(cell_coord, ncells_per_dim)
            nbr_cid = key_to_cell[nbr_key]
            nbr_proc = cell_to_proc[nbr_cid]
            if nbr_proc != self_proc:
                proc_count += 1
            % else
            for q in range(-1, 2):
                cell_coord[1] += q
            % endif
                % if obj.ndims == 2:
                nbr_key = flatten2(cell_coord, ncells_per_dim)
                nbr_cid = key_to_cell[nbr_key]
                nbr_proc = cell_to_proc[nbr_cid]
                if nbr_proc != self_proc:
                    proc_count += 1
                % elif obj.ndims == 3:
                for r in range(-1, 2):
                    cell_coord[2] += r
                    nbr_key = flatten3(cell_coord, ncells_per_dim)
                    nbr_cid = key_to_cell[nbr_key]
                    nbr_proc = cell_to_proc[nbr_cid]
                    if nbr_proc != self_proc:
                        proc_count += 1
                    cell_coord[2] -= r
                % endif
                % if obj.ndims == 2:
                cell_coord[1] -= q
                % endif
            cell_coord[0] -= p

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

    def template(self, i, cids, starts, nbr_procs, cell_to_key, ncells_per_dim,
                 key_to_cell, cell_to_proc, self_proc):
        '''
        proc_count = 0
        start = starts[i]
        cid = cids[i]
        key = cell_to_key[cid]
        cell_coord = declare('matrix(${obj.ndims}, "int")')
        unflatten${obj.ndims}(key, cell_coord, ncells_per_dim):

        for p in range(-1, 2):
            cell_coord[0] += p
            % if obj.ndims == 1:
            nbr_key = flatten1(cell_coord, ncells_per_dim)
            nbr_cid = key_to_cell[nbr_key]
            nbr_proc = cell_to_proc[nbr_cid]
            if nbr_proc != self_proc:
                nbr_procs[start + proc_count] = nbr_proc
                proc_count += 1
            % else
            for q in range(-1, 2):
                cell_coord[1] += q
            % endif
                % if obj.ndims == 2:
                nbr_key = flatten2(cell_coord, ncells_per_dim)
                nbr_cid = key_to_cell[nbr_key]
                nbr_proc = cell_to_proc[nbr_cid]
                if nbr_proc != self_proc:
                    nbr_procs[start + proc_count] = nbr_proc
                    proc_count += 1
                % elif obj.ndims == 3:
                for r in range(-1, 2):
                    cell_coord[2] += r
                    nbr_key = flatten3(cell_coord, ncells_per_dim)
                    nbr_cid = key_to_cell[nbr_key]
                    nbr_proc = cell_to_proc[nbr_cid]
                    if nbr_proc != self_proc:
                        nbr_procs[start + proc_count] = nbr_proc
                        proc_count += 1
                    cell_coord[2] -= r
                % endif
                % if obj.ndims == 2:
                cell_coord[1] -= q
                % endif
            cell_coord[0] -= p
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


class LoadBalancer(object):
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
    def __init__(self, ndims, dtype, cell_manager,
                 proc_weights=None, root=0, tag=111, padding=0.,
                 migrate=False, backend=None):
        self.comm = mpi.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.root = root
        self.backend = get_backend(backend)
        self.cm = cell_manager
        self.ndims = ndims
        self.proc_weights = proc_weights
        self.tag = tag
        self.dtype = dtype
        self.procs = None
        self.padding = padding
        self.all_gids = None
        self.lb_done = False
        self.adjusted = False
        self.migrate = migrate

        self.point_assign_f = PointAssign('point_assign', self.ndims).function
        self.point_assign_knl = Elementwise(self.point_assign_f,
                                            backend=self.backend)

        if self.rank == self.root:
            self.procs = carr.arange(0, self.size, 1, dtype=np.int32,
                                     backend=self.backend)

        if self.rank == self.root and self.proc_weights is None:
            self.proc_weights = carr.empty(self.size, np.float32,
                                           backend=self.backend)
            self.proc_weights.fill(1. / self.size)

        if self.rank == self.root:
            self.all_exec_times = np.zeros(self.size, dtype=np.float32)
            self.all_exec_nobjs = np.zeros(self.size, dtype=np.int32)
        else:
            self.all_exec_times = None
            self.all_exec_nobjs = None

    def set_cell_map(self, cell_map):
        self.cell_map = cell_map

    def _partition_procs(self):
        # NOTE: This can be made better using the weights
        # of the procs and communication costs
        part_idx = self.procs.length // 2
        target_w = carr.sum(self.proc_weights[:part_idx])
        return part_idx, target_w

    def _partition_domain(self, target_w):
        lengths = self.cm.cell_max - self.cm.cell_min

        maxlen_idx = np.argmax(lengths)

        # sort data and gids according to max length dim

        order = carr.argsort(self.cm.centroids[maxlen_idx])

        align_centroids = []
        for i, x in enumerate(self.cm.centroids):
            if i != maxlen_idx:
                align_centroids.append(x)

        inp_list = [self.cm.centroids[maxlen_idx], self.cm.cids,
                    self.cm.cell_weights] + align_centroids

        out_list = carr.sort_by_keys(inp_list)

        self.cm.cids = out_list[1]
        self.cm.cell_weights = out_list[2]

        # FIXME: Fix this mess. Perhaps by using an argument
        # in sort by keys to choose the place in the inp list
        # that has the keys
        curr_head = 3
        for i in range(self.ndims):
            if i != maxlen_idx:
                out = out_list[curr_head]
                self.cm.centroids[i] = out
                curr_head += 1

        target_idx = carr.zeros(1, dtype=np.int32, backend=self.backend)

        split_idx_knl = Scan(inp_partition_domain, out_partition_domain,
                             'a+b', dtype=self.cm.cell_weights.dtype,
                             backend=self.backend)

        split_idx_knl(weights=self.cm.cell_weights, target_w=target_w,
                      target_idx=target_idx)

        target_idx = int(target_idx[0])

        return maxlen_idx, target_idx

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
            #self.proc_weights = self.proc_weights.get()

            self.all_exec_times = np.zeros(self.size, dtype=np.float32)
            self.all_exec_nobjs = np.zeros(self.size, dtype=np.int32)

        self.adjusted = True

    def migrate_objects(self, *coords):
        new_proc = carr.empty(self.cm.num_objs, np.int32, backend=self.backend)
        new_proc.fill(self.rank)
        args = list(coords) + list(self.cm.min)
        self.point_assign_knl(self.cm.cids, new_proc, self.cm.cell_size,
                              self.cell_map.cell_proclist,
                              self.cell_map.key_to_cell, int(self.cell_map.max_key),
                              self.cm.ncells_per_dim,
                              *args)
        migrate_plan = Comm(new_proc, root=self.root, backend=self.backend)
        return migrate_plan

#    def migrate_objects(self, data):
#        new_proc = self.make_migrate_plan()
#
#        new_len = self.migrate_plan.nreturn
#        nsends = self.migrate_plan.nblocks
#        # gids, weights, coords, data
#
#        if nsends == 1 and new_len == self.num_objs and new_proc[0] == self.rank:
#            return
#
#        new_coords = []
#        new_data = []
#
#        new_gids = carr.empty(new_len, self.gids.dtype,
#                              backend=self.backend)
#        new_weights = carr.empty(new_len, self.weights.dtype,
#                                 backend=self.backend)
#        for i, x in enumerate(self.coords):
#            new_coords.append(carr.empty(new_len, x.dtype,
#                                         backend=self.backend))
#
#        for i, x in enumerate(data):
#            new_data.append(carr.empty(new_len, x.dtype, backend=self.backend))
#
#        self.migrate_plan.comm_do_post(self.gids, new_gids)
#        self.migrate_plan.comm_do_post(self.weights, new_weights)
#
#        for i, x in enumerate(self.coords):
#            self.migrate_plan.comm_do_post(x, new_coords[i])
#
#        for i, x in enumerate(data):
#            self.migrate_plan.comm_do_post(x, new_data[i])
#
#        self.migrate_plan.comm_do_wait()
#
#        self.gids.set_data(new_gids)
#        self.weights.set_data(new_weights)
#        for i, x_new in enumerate(new_coords):
#            self.coords[i].set_data(x_new)
#        for i, x_new in enumerate(new_data):
#            data[i].set_data(x_new)
#
#        self.num_objs = int(new_len)

    def load_balance_raw(self):
        # FIXME: These tags might mess with other sends and receives
        reqs_centroids = []
        req_cids = None
        req_w = None
        req_max, req_min = None, None
        req_cnobjs = None

        # Gather proc weights
        if self.proc_weights != None and self.proc_weights.length != self.size:
            curr_weight = self.proc_weights[0]

            if self.rank == self.root:
                self.proc_weights.resize(self.size)
            else:
                self.proc_weights.resize(0)

            self.comm.Gather(curr_weight, self.proc_weights.get_buff(),
                             root=self.root)

        if self.rank == self.root:
            self.procs = carr.arange(0, self.size, 1, dtype=np.int32,
                                          backend=self.backend)
        else:
            self.procs = None

        self.cm.update_cell_bounds()

        if self.rank != self.root:
            status = mpi.Status()
            nrecv_centroids = np.empty(1, dtype=np.int32)
            nrecv_procs = np.empty(1, dtype=np.int32)

            req_nprocs = self.comm.Irecv(
                    nrecv_procs, source=mpi.ANY_SOURCE, tag=self.tag + 1
                    )

            req_nprocs.Wait(status=status)
            self.parent = status.Get_source()

            req_ncentroids = self.comm.Irecv(
                    nrecv_centroids, source=self.parent, tag=self.tag + 2
                    )

            self.nrecv_procs = int(nrecv_procs)
            self.procs = carr.empty(self.nrecv_procs, np.int32,
                                    backend=self.backend)

            req_procs = self.comm.Irecv(self.procs.get_buff(),
                                        source=self.parent, tag=self.tag + 3)

            req_ncentroids.Wait()

            self.nrecv_centroids = int(nrecv_centroids)

            self.cm.cids = carr.empty(self.nrecv_centroids, np.int32,
                                      backend=self.backend)

            req_cids = self.comm.Irecv(self.cm.cids.get_buff(),
                                         source=self.parent, tag=self.tag + 4)

            self.proc_weights = carr.empty(self.nrecv_procs, self.dtype,
                                           backend=self.backend)

            req_procw = self.comm.Irecv(self.proc_weights.get_buff(),
                                        source=self.parent, tag=self.tag + 5)

            self.cm.cell_weights.resize(self.nrecv_centroids)

            req_w = self.comm.Irecv(self.cm.cell_weights.get_buff(),
                                    source=self.parent, tag=self.tag + 6)

            self.cm.cell_num_objs.resize(self.nrecv_centroids)

            req_cnobjs = self.comm.Irecv(self.cm.cell_num_objs.get_buff(),
                                         source=self.parent, tag=self.tag + 12)

            for i, x in enumerate(self.cm.centroids):
                x.resize(self.nrecv_centroids)
                reqs_centroids.append(self.comm.Irecv(x.get_buff(),
                                   source=self.parent, tag=self.tag + 7 + i))

            req_max = self.comm.Irecv(self.cm.cell_max, source=self.parent,
                                      tag=self.tag + 10)
            req_min = self.comm.Irecv(self.cm.cell_min, source=self.parent,
                                      tag=self.tag + 11)

            req_procs.Wait()
            req_procw.Wait()

            if self.procs.length == 1:
                mpi.Request.Waitall(reqs_centroids)
                req_cids.Wait()
                req_max.Wait()
                req_min.Wait()
                req_w.Wait()
                req_cnobjs.Wait()
                return

        while self.procs.length != 1:
            part_idx, target_w = self._partition_procs()

            right_proc = int(self.procs[part_idx])

            nsend_rprocs = np.asarray(self.procs.length - part_idx,
                                      dtype=np.int32)

            # transfer nsend procs
            self.comm.Send(nsend_rprocs, dest=right_proc, tag=self.tag + 1)

            if reqs_centroids:
                mpi.Request.Waitall(reqs_centroids)
                reqs_centroids = []

            if req_max:
                req_max.Wait()
                req_max = None

            if req_min:
                req_min.Wait()
                req_min = None

            if req_w:
                req_w.Wait()
                req_w = None

            if req_cids:
                req_cids.Wait()
                req_cids = None

            right_max = self.cm.cell_max.copy()
            right_min = self.cm.cell_min.copy()

            maxlen_idx, target_idx = self._partition_domain(target_w)

            part_dim = self.cm.centroids[maxlen_idx]
            part_pos = 0.5 * (part_dim[target_idx - 1] + part_dim[target_idx])

            self.cm.cell_max[maxlen_idx] = part_pos
            right_min[maxlen_idx] = part_pos

            nsend_rcentroids = np.asarray(self.cm.centroids[0].length - target_idx,
                                       dtype=np.int32)

            # transfer nsend data
            self.comm.Send(nsend_rcentroids, dest=right_proc, tag=self.tag + 2)

            # transfer procs
            self.comm.Send(self.procs.get_buff(offset=part_idx),
                           dest=right_proc, tag=self.tag + 3)

            # transfer proc weights
            self.comm.Send(self.proc_weights.get_buff(offset=part_idx),
                           dest=right_proc, tag=self.tag + 5)

            # transfer x, y, z
            for i, x in enumerate(self.cm.centroids):
                self.comm.Send(x.get_buff(offset=target_idx),
                               dest=right_proc, tag=self.tag + 7 + i)

            # transfer cell weights
            self.comm.Send(self.cm.cell_weights.get_buff(offset=target_idx),
                           dest=right_proc, tag=self.tag + 6)

            # transfer obj ids
            self.comm.Send(self.cm.cids.get_buff(offset=target_idx),
                           dest=right_proc, tag=self.tag + 4)

            self.comm.Send(right_max, dest=right_proc, tag=self.tag + 10)

            self.comm.Send(right_min, dest=right_proc, tag=self.tag + 11)

            # transfer cell num objs
            if req_cnobjs:
                req_cnobjs.Wait()
            self.comm.Send(self.cm.cell_num_objs.get_buff(offset=target_idx),
                           dest=right_proc, tag=self.tag + 12)

            self.procs.resize(part_idx)
            self.proc_weights.resize(part_idx)
            self.cm.cids.resize(target_idx)
            for x in self.cm.centroids:
                x.resize(target_idx)
            self.cm.cell_weights.resize(target_idx)
            self.cm.cell_num_objs.resize(target_idx)

    def load_balance(self):
        # NOTE: Gather before calling load balance
        self.load_balance_raw()
        return self.make_comm_plan()

    def _calculate_nobjs_displs(self, num_objs):
        if self.rank == self.root:
            gids_displs = np.zeros(self.size, np.int32)
            nobjs_per_proc = np.empty(self.size, np.int32)
        else:
            gids_displs = None
            nobjs_per_proc = None

        self.comm.Gather(np.array(num_objs, dtype=np.int32),
                         nobjs_per_proc, root=self.root)

        if self.rank == self.root:
            np.cumsum(nobjs_per_proc[:-1], out=gids_displs[1:])

        return nobjs_per_proc, gids_displs


    def make_comm_plan(self):
        # Send object ids back to root
        self.cm.num_cells = self.cm.cids.length

        old_num_objs = self.cm.num_objs

        # calculate the old gids displs before updating the num objs

        if self.lb_done:
            old_nobjs_per_proc, old_gids_displs = \
                    self._calculate_nobjs_displs(self.cm.num_objs)

        self.cm.calculate_num_objs()

        nobjs_per_proc, gids_displs = \
                self._calculate_nobjs_displs(self.cm.num_objs)

        ncells_per_proc, cell_displs = \
                self._calculate_nobjs_displs(self.cm.num_cells)

        if self.rank == self.root:
            self.total_num_objs = np.sum(nobjs_per_proc, dtype=np.int32)
            self.total_num_cells = np.sum(ncells_per_proc, dtype=np.int32)

        if self.rank == self.root:
            if self.lb_done:
                old_all_gids = self.all_gids
                old_all_gids_buff = old_all_gids.get_buff()

            self.all_gids = carr.empty(self.total_num_objs, np.int32,
                                       backend=self.backend)
            self.all_cids = carr.empty(self.total_num_cells, np.int32,
                                       backend=self.backend)
            self.all_cell_nobjs = carr.empty(self.total_num_cells, np.int32,
                                                backend=self.backend)
            all_cids_buff = self.all_cids.get_buff()
            all_cell_nobjs_buff = self.all_cell_nobjs.get_buff()
        else:
            old_all_gids = None
            self.all_gids = None
            old_all_gids_buff = None
            all_cids_buff = None
            all_cell_nobjs_buff = None

        if self.lb_done:
            # Gather old all gids
            self.comm.Gatherv(sendbuf=[self.cm.gids.get_buff(),
                                       dtype_to_mpi(self.cm.gids.dtype)],
                              recvbuf=[old_all_gids_buff, old_nobjs_per_proc,
                                       gids_displs, dtype_to_mpi(self.cm.gids.dtype)],
                              root=self.root)

        # Make a cell map
        self.comm.Gatherv(sendbuf=[self.cm.cids.get_buff(),
                                      dtype_to_mpi(self.cm.cids.dtype)],
                             recvbuf=[all_cids_buff, ncells_per_proc, cell_displs,
                                      dtype_to_mpi(self.cm.cids.dtype)],
                             root=self.root)

        self.comm.Gatherv(sendbuf=[self.cm.cell_num_objs.get_buff(),
                                      dtype_to_mpi(self.cm.cell_num_objs.dtype)],
                             recvbuf=[all_cell_nobjs_buff, ncells_per_proc, cell_displs,
                                      dtype_to_mpi(self.cm.cids.dtype)],
                             root=self.root)

        if self.migrate:
            if self.rank == self.root:
                cell_proclist = carr.empty(self.total_num_cells, np.int32,
                                           backend=self.backend)

                fill_new_proclist_knl = get_elwise(fill_new_proclist,
                                                   backend=self.backend)

                cell_displs_arr = wrap(cell_displs, backend=self.backend)

                fill_new_proclist_knl(cell_proclist, cell_displs_arr,
                                      self.size)

                self.cell_map.update(self.all_cids, cell_proclist, self.cm.cell_to_key,
                                     self.cm.max_key)

            self.cell_map.bcast()

        #if self.rank == self.root:
        #    self.all_cell_nobjs = self.all_cell_nobjs.align(self.all_cids)
        #    fill_gids_from_cids_knl = get_scan(inp_fill_gids_from_cids,
        #                                       out_fill_gids_from_cids,
        #                                       np.int32, self.backend)
        #    fill_gids_from_cids_knl(all_cids=self.all_cids,
        #                            all_gids=self.all_gids,
        #                            gids=self.cm.gids,
        #                            cell_to_idx=self.cm.cell_to_idx,
        #                            cell_num_objs=self.all_cell_nobjs)

        if not self.lb_done:
            plan = CommBase(None, sorted=True, root=self.root, tag=self.tag)
            procs_from = np.array([0], dtype=np.int32)
            lengths_from = np.array([self.cm.num_objs], dtype=np.int32)

            #dbg_print("%s %s" % (procs_from, lengths_from))

            if self.rank == self.root:
                procs_to = np.arange(0, self.size, dtype=np.int32)
                lengths_to = nobjs_per_proc
            else:
                procs_to = np.array([], dtype=np.int32)
                lengths_to = np.array([], dtype=np.int32)

            plan.set_send_info(procs_to, lengths_to)
            plan.set_recv_info(procs_from, lengths_from)
        else:
            if self.rank == self.root:
                new_proclist = carr.empty(self.total_num_objs, np.int32,
                                          backend=self.backend)

                fill_new_proclist_knl = get_elwise(fill_new_proclist,
                                                   backend=self.backend)

                # FIXME: Avoid wrapping this
                gids_displs_arr = wrap(gids_displs, backend=self.backend)
                fill_new_proclist_knl(new_proclist, gids_displs_arr, self.size)

                semi_aligned_proclist = new_proclist.align(self.all_gids)

                semi_aligned_proclist.align(old_all_gids, out=new_proclist)

                new_proclist_buff = new_proclist.get_buff()
            else:
                new_proclist_buff = None

            self.transfer_proclist = carr.empty(old_num_objs,
                                                np.int32, backend=self.backend)

            self.comm.Scatterv(sendbuf=[new_proclist_buff,
                                        old_nobjs_per_proc, old_gids_displs,
                                        dtype_to_mpi(np.int32)],
                               recvbuf=[self.transfer_proclist.get_buff(),
                                        dtype_to_mpi(np.int32)],
                               root=self.root)

            plan = Comm(self.transfer_proclist, backend=self.backend)

        # transfer gids using the comm plan
        new_gids = carr.empty(plan.nreturn, np.int32, backend=self.backend)
        plan.comm_do(self.cm.gids, new_gids)
        # NOTE: The old gids array can be reused
        self.cm.gids = new_gids

        # IMP FIXME: FIX THIS
        # self.lb_done = True
        return plan