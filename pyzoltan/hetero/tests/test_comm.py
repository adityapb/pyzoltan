import unittest
from pytest import mark, importorskip

from pyzoltan.tools import run_parallel_script

path = run_parallel_script.get_directory(__file__)

class HeteroTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        importorskip("mpi4py.MPI")

    @mark.parallel
    def test_hetero_comm(self):
        run_parallel_script.run(
            filename='hetero_comm.py', nprocs=2, path=path
        )


