"""Tests for the Zoltan unstructured communication package"""
import mpi4py.MPI as mpi
import numpy as np
from numpy import random
from pyzoltan.hetero.comm import Comm
from compyle.array import wrap_array
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
atexit.register(mpi.Finalize)

# each processor creates some random data
numObjectsTotal = 1 << 10

x = random.random(numObjectsTotal)
gids = np.array(np.arange(size * numObjectsTotal)
                )[rank * numObjectsTotal:(rank + 1) * numObjectsTotal]
gids = gids.astype(np.uint32)

# arbitrarily assign some objects to be sent to some other processor
nsend = np.int32(random.random_integers(low=1, high=5))
object_ids = random.random_integers(low=0, high=numObjectsTotal, size=nsend)
proclist = random.random_integers(
    low=0, high=size - 1, size=nsend
).astype(np.int32)

my_indices = np.where(proclist == rank)[0]
proclist[my_indices] = (rank + 1) % size

# create the ZComm object
tag = np.int32(0)
hcomm = Comm(proclist, tag=tag, backend='cuda')

# the data to send and receive
senddata = x[object_ids]
recvdata = np.ones(hcomm.nreturn)

# use zoltan to exchange doubles
print("Proc %d, Sending %s to %s" % (rank, senddata, proclist))
senddata = wrap_array(senddata, backend='cuda')
recvdata = wrap_array(recvdata, backend='cuda')
hcomm.comm_do(senddata, recvdata)
print("Proc %d, Received %s" % (rank, recvdata))

# use zoltan to exchange unsigned ints
senddata = gids[object_ids]
recvdata = np.ones(hcomm.nreturn, dtype=np.uint32)

print("Proc %d, Sending %s to %s" % (rank, senddata, proclist))
senddata = wrap_array(senddata, backend='cuda')
recvdata = wrap_array(recvdata, backend='cuda')
hcomm.comm_do(senddata, recvdata)
print("Proc %d, Received %s" % (rank, recvdata))

# Test the Comm Reverse function
# modify the received data
#recvdata[:] = rank
#
#updated_info = np.zeros(zcomm.nsend, dtype=senddata.dtype)
#print('Proc %d, sending updated data %s' % (rank, recvdata))
#zcomm.Comm_Do_Reverse(recvdata, updated_info)
#print('Proc %d, received updated data %s' % (rank, updated_info))