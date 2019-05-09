"""Tests for the Zoltan unstructured communication package"""
import numpy as np
from numpy import random
import mpi4py.MPI as mpi
from pyzoltan.hetero.comm import Comm
from compyle.array import wrap_array
import compyle.array as carr
#from pyzoltan.hetero.rcb import RCB
from pyzoltan.hetero.load_balancer import LoadBalancer

# MPI comm, rank and size
comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    numObjectsTotal = 2000

    print("Total num objects = %s" % numObjectsTotal)

    x = random.random(numObjectsTotal)
    y = random.random(numObjectsTotal)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    gids = np.arange(numObjectsTotal)
    gids = gids.astype(np.int32)
    gids = wrap_array(gids, backend='cython')
    x = wrap_array(x, backend='cython')
    y = wrap_array(y, backend='cython')

    proc_weights = carr.empty(size, np.float32, backend='cython')

    proc_weights[0] = 0.5
    proc_weights[1] = 0.3
    proc_weights[2] = 0.1
    proc_weights[3] = 0.05
    proc_weights[4] = 0.05
else:
    x, y, gids = None, None, None
    proc_weights = None

lb = LoadBalancer(2, np.float32)
lb_data = lb.lb_data
lb.set_coords(x=x, y=y)
lb.set_gids(gids)
lb.set_proc_weights(proc_weights)

lb.load_balance()

print("After lb, rank %s = %s" % (rank, lb_data.num_objs))

print(rank, lb.lb_obj.min, lb.lb_obj.max)

if rank == 0:
    import matplotlib.pyplot as plt
    for i in range(size):
        ids = lb.lb_obj.all_gids[lb.lb_obj.plan.proclist == i]
        plt.scatter(x[ids], y[ids], c=np.random.rand(3, ))

    plt.show()
