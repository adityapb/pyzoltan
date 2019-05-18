from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import mpi4py.MPI as mpi
import numpy as np
import sys


def only_root(f):
    if not mpi.Is_initialized():
        mpi.Init()
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    def wrapper(*args, **kwargs):
        if rank == 0:
            f(*args, **kwargs)
    return wrapper


def is_using_ipython():
    """Return True if the code is being run from an IPython session or
    notebook.
    """
    try:
        # If this is being run inside an IPython console or notebook
        # then this is defined.
        __IPYTHON__
    except NameError:
        return False
    else:
        return True


class ParallelManager(object):
    def __init__(self):
        if not mpi.Is_initialized():
            mpi.Init()
        self.comm = mpi.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.ctx = None
        self.args = sys.argv[1:]
        self._setup_argparse()
        self._parse_command_line(self.args)
        self._setup_backend()
        if self.backend == 'cuda':
            self._setup_devices()

    def __del__(self):
        if self.ctx:
            self.ctx.pop()

    def _setup_argparse(self):
        usage = '%(prog)s [options]'
        description = ""
        parser = ArgumentParser(
            usage=usage,
            description=description,
            formatter_class=ArgumentDefaultsHelpFormatter)
        self.arg_parse = parser


        # --logfile

        parser.add_argument(
            "--ncpu",
            action="store",
            type=int,
            dest="ncpu",
            default=self.size,
            help="Number of CPU processes")

        parser.add_argument(
            "--ngpu",
            action="store",
            type=int,
            dest="ngpu",
            default=0,
            help="Number of GPU processes")

    def _parse_command_line(self, args):
        self.options = self.arg_parse.parse_args(self.args)

    def _setup_backend(self):
        if self.rank % 2 == 0 and self.rank // 2 < self.options.ngpu:
            self.backend = 'cuda'
            self.device_id = self.rank // 2
        else:
            alloc_ngpu = int(np.ceil(self.size / 2.))
            if alloc_ngpu >= self.options.ngpu:
                self.backend = 'cython'
            else:
                extra_gpu = self.options.ngpu - alloc_ngpu
                if (self.rank + 1) // 2 <= extra_gpu:
                    self.backend = 'cuda'
                    self.device_id = alloc_ngpu - 1 + (self.rank + 1) // 2

    def _setup_devices(self):
        import pycuda.driver as drv
        import compyle.cuda as cu
        drv.init()
        self.dev = drv.Device(self.device_id)
        self.ctx = self.dev.make_context()
        cu.cuda_ctx = True
