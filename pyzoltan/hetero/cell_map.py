from compyle.parallel import Elementwise
from compyle.types import annotate
from compyle.array import get_backend
from pyzoltan.hetero.comm import dbg_print
import compyle.array as carr
import numpy as np
import mpi4py.MPI as mpi


@annotate
def map_key_to_cell(i, cell_to_key, key_to_cell):
    key = cell_to_key[i]
    key_to_cell[key] = i


class CellMap(object):
    def __init__(self, backend, root=0):
        # set backend
        self.backend = get_backend(backend)
        self.comm = mpi.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.root = root

        # allocate all arrays
        self.cell_proclist = carr.empty(1, np.int32, backend=self.backend)
        self.key_to_cell = carr.empty(1, np.int32, backend=self.backend)
        self.cell_to_key = carr.empty(1, np.int64, backend=self.backend)
        self.max_key = np.empty(1, dtype=np.int64)
        self.all_cids = None

    def resize(self, num_cells, max_key):
        if self.cell_proclist.length != num_cells:
            self.cell_proclist.resize(num_cells)

        if self.max_key != self.key_to_cell.length:
            self.key_to_cell.resize(1 + max_key)

        if not self.all_cids or num_cells != self.all_cids.length:
            self.all_cids = carr.arange(0, num_cells, 1, np.int32,
                                        backend=self.backend)

    def bcast(self, total_num_cells):
        self.comm.Bcast(self.max_key, root=self.root)
        self.resize(total_num_cells, int(self.max_key))
        self.cell_to_key.resize(total_num_cells)

        self.comm.Bcast(self.cell_proclist.get_buff(), root=self.root)
        self.comm.Bcast(self.key_to_cell.get_buff(), root=self.root)
        self.comm.Bcast(self.cell_to_key.get_buff(), root=self.root)

    def update(self, all_cids, cell_proclist, cell_to_key, max_key):
        self.max_key = np.array(max_key, dtype=np.int64)
        self.cell_to_key = cell_to_key
        self.resize(cell_proclist.length, int(self.max_key))

        self.all_cids, self.cell_proclist = carr.sort_by_keys([all_cids, cell_proclist])

        # invert cell to key map
        # fill key to cell map
        map_key_to_cell_knl = Elementwise(map_key_to_cell,
                                          backend=self.backend)
        map_key_to_cell_knl(self.cell_to_key, self.key_to_cell)

