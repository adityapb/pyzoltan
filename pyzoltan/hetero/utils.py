import pycuda.driver as drv
import compyle.cuda as cu
import mpi4py.MPI as mpi
import cProfile


def make_context(device_id=None, backend='cuda'):
    if backend != 'cuda':
        return
    if device_id is None:
        comm = mpi.COMM_WORLD
        device_id = comm.Get_rank()
    ctx = cu.set_context(device_id)
    return ctx


def reduce_time(t):
    comm = mpi.COMM_WORLD
    return comm.allreduce(t, op=max)


def profile(filename=None, comm=mpi.COMM_WORLD):
  def prof_decorator(f):
    def wrap_f(*args, **kwargs):
      pr = cProfile.Profile()
      pr.enable()
      result = f(*args, **kwargs)
      pr.disable()

      if filename is None:
        pr.print_stats()
      else:
        filename_r = filename + ".{}".format(comm.rank)
        pr.dump_stats(filename_r)

      return result
    return wrap_f
  return prof_decorator


