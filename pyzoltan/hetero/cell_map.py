from compyle.parallel import Elementwise
from compyle.types import annotate
from compyle.array import get_backend
import compyle.array as carr


@annotate
def map_key_to_cell(i, cell_to_key, key_to_cell):
    key = cell_to_key[i]
    key_to_cell[key] = i


class CellMap(object):
    def __init__(self, backend):
        # set backend
        self.backend = get_backend(backend)
        # allocate all arrays
        self.cell_proclist = carr.empty(1, np.int32, backend=self.backend)
        self.key_to_cell = carr.empty(1, np.int32, backend=self.backend)
        self.all_cids = None

    def update(self, all_cids, cell_proclist, cell_to_key, max_key):
        if self.cell_proclist.length != cell_proclist.length:
            self.cell_proclist.resize(cell_proclist.length)

        cell_proclist.align(all_cids, out=self.cell_proclist)

        if self.all_cids or all_cids.length != self.all_cids.length:
            self.all_cids = carr.arange(0, all_cids.length, np.int32,
                                        backend=self.backend)

        # invert cell to key map
        if max_key != self.key_to_cell.length:
            self.key_to_cell.resize(max_key)

        # fill key to cell map
        map_key_to_cell_knl = Elementwise(map_key_to_cell,
                                          backend=self.backend)
        map_key_to_cell_knl(self.cell_to_key, self.key_to_cell)
