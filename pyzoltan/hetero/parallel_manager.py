from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import mpi4py.MPI as mpi


def only_root(f):
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
        self.comm = mpi.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.ctx = None
        self.args = sys.argv[1:]
        self._setup_argparse()
        self._parse_command_line(args)
        if self.options.ngpu:
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
            dest="ncpu",
            default=self.size,
            help="Number of CPU processes")

        parser.add_argument(
            "--ngpu",
            action="store",
            dest="ngpu",
            default=0,
            help="Number of GPU processes")

    def _parse_command_line(self, args):
        self.options = self.arg_parse.parse_args(self.args)

    def _setup_backend(self):
        if self.rank < self.options.ncpu:
            self.backend = 'cython'
        else:
            self.backend = 'cuda'

    def _setup_devices(self):
        import pycuda.driver as drv
        drv.init()
        self.dev = drv.Device(self.rank - self.options.ncpu)
        self.ctx = self.dev.make_context()
