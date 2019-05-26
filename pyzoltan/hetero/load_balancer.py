from pyzoltan.hetero.comm import Comm, dtype_to_mpi
from pyzoltan.hetero.rcb import RCB, dbg_print
from compyle.array import get_backend
from pytools.decorator import decorator
import compyle.array as carr
import mpi4py.MPI as mpi
import time
import numpy as np


def adapt_lb(**kwargs):
    lbalancer = kwargs.get('lb_obj', None)
    lb_name = kwargs.get('lb_name', None)
    if not lbalancer:
        @decorator
        def timer(f, *args, **kwargs):
            start = time.time()
            ret_val = f(*args, **kwargs)
            end = time.time()
            lbalancer = getattr(args[0], lb_name)
            if lbalancer.exec_count == 0:
                lbalancer.exec_count += 1
            else:
                lbalancer.exec_time += end - start
                lbalancer.exec_nobjs += lbalancer.lb_obj.num_objs
                lbalancer.exec_count += 1
            return ret_val
    else:
        @decorator
        def timer(f, *args, **kwargs):
            start = time.time()
            ret_val = f(*args, **kwargs)
            end = time.time()
            if lbalancer.exec_count == 0:
                lbalancer.exec_count += 1
            else:
                lbalancer.exec_time += end - start
                lbalancer.exec_nobjs += lbalancer.lb_obj.num_objs
                lbalancer.exec_count += 1
            return ret_val
    return timer


class LoadBalancerData(object):
    def __init__(self):
        pass


class LoadBalancer(object):
    def __init__(self, ndims, dtype, **kwargs):
        if not mpi.Is_initialized():
            mpi.Init()
        self.algorithm = kwargs.get('algorithm', RCB)
        self.root = kwargs.get('root', 0)
        self.backend = get_backend(kwargs.get('backend', None))
        self.ndims = ndims
        self.dtype = dtype
        self.exec_time = 0.
        self.exec_nobjs = 0
        self.exec_count = 0
        self.lb_obj = None
        self.lb_count = 0

        self.coords = []
        self.proc_weights = None
        self.weights = None
        self.gids = None
        self.data = []

        self.lb_data = LoadBalancerData()

    def set_coords(self, **kwargs):
        self.coord_names = list(kwargs.keys())
        self.coords = list(kwargs.values())
        self.coords = [] if all(v is None for v in self.coords) else \
                        self.coords
        if self.coords and self.coords[0].dtype != self.dtype:
            raise TypeError("dtype %s doesn't match with coords dtype %s" % \
                    (self.dtype, self.coords[0].dtype))

    def set_proc_weights(self, proc_weights):
        self.proc_weights = proc_weights

    def set_weights(self, weights):
        self.weights = weights

    def set_gids(self, gids):
        self.gids = gids

    def set_data(self, **kwargs):
        self.data_names = list(kwargs.keys())
        self.data = list(kwargs.values())
        self.data = [] if all(v is None for v in self.data) else \
                        self.data
        self.local_data = [None] * len(self.data_names)

    def set_lbfreq(self, lbfreq):
        self.lbfreq = lbfreq

    def add_data(self, ary, name):
        self.data_names.append(name)
        self.data.append(ary)

    def update_lb_data(self, data=None):
        if not data:
            data = self.local_data

        for i, x in enumerate(data):
            setattr(self.lb_data, self.data_names[i], x)

        for i, x in enumerate(self.lb_obj.coords_view):
            setattr(self.lb_data, self.coord_names[i], x)

        setattr(self.lb_data, 'gids', self.lb_obj.gids_view)
        setattr(self.lb_data, 'weights', self.lb_obj.weights_view)
        setattr(self.lb_data, 'proc_weights', self.lb_obj.proc_weights_view)
        setattr(self.lb_data, 'num_objs', self.lb_obj.num_objs)

    def gather(self):
        gath_data = self.lb_obj.gather(self.local_data)
        self.update_lb_data(data=gath_data)

    def load_balance(self):
        if not self.lb_obj:
            self.lb_obj = self.algorithm(self.ndims, self.dtype, coords=self.coords,
                                     gids=self.gids, weights=self.weights,
                                     proc_weights=self.proc_weights,
                                     root=self.root, backend=self.backend)

        if self.lb_count:
            self.lb_obj.gather(self.local_data)

        if self.exec_count:
            self.lb_obj.adjust_proc_weights(self.exec_time, self.exec_nobjs,
                                        self.exec_count)

        self.lb_obj.load_balance()

        len_data = len(self.data_names)

        if self.lb_obj.rank == self.lb_obj.root and self.data:
            # FIXME: This is creating new arrays everytime
            aligned_data = carr.align(self.data, self.lb_obj.all_gids,
                                      backend=self.backend)
        else:
            aligned_data = [None] * int(len_data)

        senddtype = self.data[0].dtype if self.data else None

        recvdtype = self.lb_obj.plan.get_recvdtype(senddtype)

        for i, senddata in enumerate(aligned_data):
            if recvdtype:
                recvdata = carr.empty(self.lb_obj.plan.nreturn, dtype=recvdtype,
                                      backend=self.backend)
            else:
                recvdata = None
            self.lb_obj.comm_do(senddata, recvdata)
            self.local_data[i] = recvdata

        self.update_lb_data()

        self.exec_time = 0.
        self.exec_nobjs = 0
        self.exec_count = 0
        self.lb_count += 1

    def update(self, migrate=True):
        if self.lb_count % self.lbfreq == 0:
            self.load_balance()
            self.update_lb_data()
        elif migrate:
            data = [getattr(self.lb_data, x) for x in self.data_names]
            self.lb_obj.migrate_objects(data)
            self.update_lb_data()

