import numpy as np


def CellManager(object):
    def __init__(self, ndims, cell_size):
        self.ndims = ndims
        self.cell_size = cell_size

    def generate_cells(self, coords, weights):
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


