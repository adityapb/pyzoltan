import unittest
from pytest import mark, importorskip
from pyzoltan.hetero.cell_manager import CellManager
from numpy import random
import compyle.array as carr
import numpy as np

class TestCellManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        importorskip("mpi4py.MPI")

    def test_cell_generator(self):
        cell_manager = CellManager(1, np.float32, 0.1, num_objs=100)
        x = random.random(100).astype(np.float32)
        x = carr.wrap(x, backend='cython')

        weights = carr.empty(100, np.float32, backend='cython')
        weights.fill(1. / 100)

        gids = carr.arange(0, 100, 1, np.int32, backend='cython')

        cell_manager.set_coords([x])
        cell_manager.set_weights(weights)
        cell_manager.set_gids(gids)

        cell_manager.update_bounds()

        cell_manager.generate_cells()

        self.assertEqual(sum(cell_manager.cell_num_objs.get()), 100)
        self.assertAlmostEqual(sum(cell_manager.cell_weights.get()), 1.)
