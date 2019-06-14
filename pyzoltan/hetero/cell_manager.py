import numpy as np
from compyle.template import Template
from compyle.parallel import Elementwise, Scan, Reduction
from compyle.types import annotate, declare
from compyle.low_level import cast
from compyle.array import get_backend
from pytools import memoize
from pyzoltan.hetero.comm import get_elwise, get_scan
import compyle.array as carr
from compyle.array import wrap


@annotate(point='intp', ncells_per_dim='intp', return_='long')
def flatten1(point, ncells_per_dim):
    res = declare('long')
    res = point[0]
    return res


@annotate(point='intp', ncells_per_dim='intp', return_='long')
def flatten2(point, ncells_per_dim):
    ncx = ncells_per_dim[0]
    res = declare('long')
    res = point[0] + ncx * point[1]
    return res


@annotate(point='intp', ncells_per_dim='intp', return_='long')
def flatten3(point, ncells_per_dim):
    ncx = ncells_per_dim[0]
    ncy = ncells_per_dim[1]
    res = declare('long')
    res = point[0] + ncx * point[1] + ncx * ncy * point[2]
    return res


@annotate
def unflatten1(key, cell_coord, ncells_per_dim):
    cell_coord[0] = cast(key, 'int')


@annotate
def unflatten2(key, cell_coord, ncells_per_dim):
    ncx = ncells_per_dim[0]
    cell_coord[1] = cast(key / ncx, 'int')
    cell_coord[0] = cast(key - cell_coord[1] * ncx, 'int')


@annotate
def unflatten3(key, cell_coord, ncells_per_dim):
    ncx = ncells_per_dim[0]
    ncy = ncells_per_dim[1]
    cell_coord[2] = cast(key / (ncx * ncy), 'int')
    cell_coord[1] = cast((key - cell_coord[2] * ncx * ncy) / ncx, 'int')
    cell_coord[0] = cast(key - cell_coord[1] * ncx - \
            cell_coord[2] * ncy * ncx, 'int')


class FillKeys(Template):
    def __init__(self, name, ndims):
        super(FillKeys, self).__init__(name=name)
        self.ndims = ndims
        self.args = []
        for i in range(self.ndims):
            self.args.append('x%s' % i)

    def extra_args(self):
        min_args = ['%smin' % arg for arg in self.args]
        return self.args + min_args, {}

    def template(self, i, keys, cell_size, ncells_per_dim):
        '''
        point = declare('matrix(${obj.ndims}, "int")')
        % for j, x in enumerate(obj.args):
        point[${j}] = cast(floor((${x}[i] - ${x}min) / cell_size), 'int')
        % endfor
        keys[i] = flatten${obj.ndims}(point, ncells_per_dim)
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
        return self.args + min_args, {}

    def template(self, i, item, prev_item, ncells_per_dim, keys,
                 cell_size, cell_to_key, cell_to_idx, key_to_idx):
        # FIXME: key auto declared incorrectly
        '''
        cids = declare('matrix(${obj.ndims}, "int")')
        key = declare('long')
        if i == 0 or item != prev_item:
            key = keys[i]
            unflatten${obj.ndims}(key, cids, ncells_per_dim)
            % for dim in range(obj.ndims):
            centroid${dim}[item] = x${dim}min + cell_size * (0.5 + cids[${dim}])
            % endfor
            cell_to_key[item] = key
            cell_to_idx[item] = i
            key_to_idx[key] = i
        '''


@annotate
def fill_cell_weights(i, cell_to_idx, cell_weights, cell_num_objs, weights,
                      num_cells, total_num_objs):
    start = cell_to_idx[i]
    if i < num_cells - 1:
        end = cell_to_idx[i + 1]
    else:
        end = total_num_objs
    num_objs = end - start

    tot_weight = 0.
    for j in range(num_objs):
        tot_weight += weights[start + j]

    cell_num_objs[i] = num_objs
    cell_weights[i] = tot_weight


@memoize(key=lambda *args: tuple(args))
def get_fill_centroids_kernel(ndims, backend):
    out_fill_centroids = OutFillCentroids(
            'out_fill_centroids', ndims
            ).function
    fill_centroids_knl = get_scan(inp_fill_centroids, out_fill_centroids,
                                  np.int32, backend)
    return fill_centroids_knl


@annotate
def find_num_objs(i, cids, cell_num_objs):
    cid = cids[i]
    return cell_num_objs[cid]


class CellManager(object):
    def __init__(self, ndims, dtype, cell_size, padding=0., num_objs=0,
                 backend=None):
        self.backend = get_backend(backend)
        self.ndims = ndims
        self.num_objs = num_objs
        self.cell_size = cell_size
        self.padding = padding
        self.dtype = dtype

        self.max = np.empty(ndims, dtype=dtype)
        self.min = np.empty(ndims, dtype=dtype)

        self.cell_max = np.empty(ndims, dtype=dtype)
        self.cell_min = np.empty(ndims, dtype=dtype)

        self.keys = carr.empty(self.num_objs, np.int64,
                               backend=self.backend)
        self.key_to_idx = carr.zeros(1, np.int32,
                                     backend=self.backend)

        self.cell_to_key = carr.empty(1, np.int64,
                                      backend=self.backend)
        self.cell_to_idx = carr.empty(1, np.int32,
                                      backend=self.backend)
        self.cell_weights = carr.empty(1, np.float32,
                                       backend=self.backend)

        self.ncells_per_dim = carr.empty(self.ndims, np.int32,
                                         backend=self.backend)

        self.centroids = []
        for dim in range(self.ndims):
            self.centroids.append(carr.empty(1, self.dtype,
                                             backend=self.backend))

        self.cell_num_objs = carr.empty(1, np.int32, backend=self.backend)

    def set_coords(self, coords):
        self.coords = coords

    def set_weights(self, weights):
        if weights:
            self.weights = weights
        else:
            self.weights = carr.empty(self.num_objs, np.float32,
                                      backend=self.backend)

    def set_gids(self, gids):
        if gids:
            self.gids = gids
        else:
            self.gids = carr.arange(0, self.num_objs, 1, np.int32,
                                    backend=self.backend)

    def update_bounds(self):
        if self.coords:
            carr.update_minmax(self.coords)

        for i, x in enumerate(self.coords):
            xlength = x.maximum - x.minimum
            eps = 0.
            if xlength == 0:
                eps = 10 * np.finfo(np.float32).eps
            self.max[i] = eps + x.maximum + self.padding * xlength
            self.min[i] = -eps + x.minimum - self.padding * xlength

    def update_cell_bounds(self):
        if self.centroids:
            carr.update_minmax(self.centroids)

        for i, x in enumerate(self.centroids):
            xlength = x.maximum - x.minimum
            eps = 0.
            if xlength == 0:
                eps = 10 * np.finfo(np.float32).eps
            self.cell_max[i] = eps + x.maximum + self.padding * xlength
            self.cell_min[i] = -eps + x.minimum - self.padding * xlength


    def generate_cells(self):
        #if self.rank == self.root and self.weights is None:
        #    self.weights = carr.empty(self.num_objs, np.float32,
        #                              backend=self.backend)
        #    self.weights.fill(1. / self.num_objs)
        #    self.gids = carr.arange(0, self.num_objs, 1, np.int32,
        #                            backend=self.backend)
        #elif self.rank == self.root:
        #    self.weights.dev /= carr.sum(self.weights)

        fill_keys_knl = get_fill_keys_kernel(self.ndims, self.backend)
        fill_centroids_knl = get_fill_centroids_kernel(self.ndims,
                                                       self.backend)
        fill_cell_weights_knl = get_elwise(fill_cell_weights,
                                           backend=self.backend)

        ncells_per_dim = np.empty(self.ndims, dtype=np.int32)
        for i in range(self.ndims):
            ncells_per_dim[i] = \
                    np.ceil((self.max[i] - self.min[i]) / self.cell_size)

        self.ncells_per_dim.set(ncells_per_dim)

        self.keys.resize(self.gids.length)

        fill_keys_args = [self.keys, self.cell_size, self.ncells_per_dim] + \
                self.coords + list(self.min)

        fill_keys_knl(*fill_keys_args)

        # sort
        inp_list = [self.keys, self.gids] + self.coords
        out_list = carr.sort_by_keys(inp_list, backend=self.backend)
        self.keys, self.gids = out_list[0], out_list[1]
        self.coords = out_list[2:]

        self.max_key = int(self.keys[-1])

        self.key_to_idx.resize(1 + self.max_key)
        self.key_to_idx.fill(0)

        num_cells_knl = Reduction('a+b', map_func=inp_fill_centroids,
                                 dtype_out=np.int32, backend=self.backend)

        self.num_cells = int(num_cells_knl(self.keys))

        self.cell_to_key.resize(self.num_cells)
        self.cell_to_idx.resize(self.num_cells)
        self.cell_weights.resize(self.num_cells)

        for x in self.centroids:
            x.resize(self.num_cells)

        fill_centroids_args = {'keys': self.keys,
                               'ncells_per_dim': self.ncells_per_dim,
                               'cell_size': self.cell_size,
                               'cell_to_key': self.cell_to_key,
                               'key_to_idx': self.key_to_idx,
                               'cell_to_idx': self.cell_to_idx}

        for dim in range(self.ndims):
            arg_name = 'x%s' % dim
            min_arg_name = 'x%smin' % dim
            cen_arg_name = 'centroid%s' % dim
            fill_centroids_args[arg_name] = self.coords[dim]
            fill_centroids_args[min_arg_name] = self.min[dim]
            fill_centroids_args[cen_arg_name] = self.centroids[dim]

        fill_centroids_knl(**fill_centroids_args)

        self.cids = carr.arange(0, self.num_cells, 1, np.int32,
                                backend=self.backend)

        self.cell_num_objs.resize(self.num_cells)

        fill_cell_weights_knl(self.cell_to_idx, self.cell_weights,
                              self.cell_num_objs, self.weights,
                              self.num_cells, self.num_objs)

    def calculate_num_objs(self):
        find_num_objs_knl = get_reduction(find_num_objs)
        self.num_objs = find_num_objs_knl(self.cids, self.cell_num_objs)
