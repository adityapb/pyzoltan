"""Tests for the Zoltan unstructured communication package"""
import mpi4py.MPI as mpi
import numpy as np
from numpy import random
from pyzoltan.hetero.comm import Comm
from pyzoltan.hetero.rcb import RCB
from compyle.array import wrap_array
import compyle.array as carr
import pycuda.driver as drv
import atexit

drv.init()

# MPI comm, rank and size
comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

dev = drv.Device(rank)
ctx = dev.make_context()
atexit.register(ctx.pop)

if rank == 0:
    numObjectsTotal = 1 << 10

    print("Total num objects = %s" % numObjectsTotal)

    x = random.random(numObjectsTotal)
    x = x.astype(np.float32)
    gids = np.arange(numObjectsTotal)
    gids = gids.astype(np.uint32)
    gids = wrap_array(gids, backend='cuda')
    x = wrap_array(x, backend='cuda')

    proc_weights = carr.empty(2, np.float32, backend='cuda')

    proc_weights[0] = 0.45
    proc_weights[1] = 0.55

if rank == 0:
    rcb = RCB(1, np.float32, data=[x], object_ids=gids, backend='cuda',
              proc_weights=proc_weights)
else:
    rcb = RCB(1, np.float32, backend='cuda')

rcb.load_balance()

print("After lb, rank %s = %s" % (rank, rcb.object_ids_view.length))

