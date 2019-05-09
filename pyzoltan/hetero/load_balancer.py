from pyzoltan.hetero.comm import Comm, dtype_to_mpi
from pyzoltan.hetero.rcb import RCB
from compyle.array import get_backend


def adapt_lb(**kwargs):
    lb_obj = kwargs.get('lb_obj', None)
    def timer(f):
        def wrapper(*args, **kwargs):
            start = time.time()
            ret_val = f(*args, **kwargs)
            end = time.time()
            if not lb_obj:
                lb_obj = getattr(args[0], kwargs.get('lb_name'))
            lb_obj.exec_time += end - start
            lb_obj.exec_nobjs += lb_obj.lb.num_objs
            lb_obj.exec_count += 1
            return ret_val
        return wrapper
    return timer


class LoadBalancerData(object):
    def __init__(self):
        pass


class LoadBalancer(object):
    def __init__(self, ndims, dtype, **kwargs):
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

        self.coords = None
        self.proc_weights = None
        self.weights = None
        self.gids = None
        self.data = None

        self.lb_data = LoadBalancerData()

    def set_coords(self, **kwargs):
        self.coord_names = list(kwargs.keys())
        self.coords = list(kwargs.values())
        self.coords = None if all(v is None for v in self.coords) else \
                        self.coords

    def set_proc_weights(self, proc_weights):
        self.proc_weights = proc_weights

    def set_weights(self, weights):
        self.weights = weights

    def set_gids(self, gids):
        self.gids = gids

    def set_data(self, **kwargs):
        self.data_names = list(kwargs.keys())
        self.data = list(kwargs.values())
        self.data = None if all(v is None for v in self.data) else \
                        self.data

    def add_data(self, ary, name):
        self.data_names.append(name)
        self.data.append(ary)

    def load_balance(self):
        if not self.lb_obj:
            self.lb_obj = self.algorithm(self.ndims, self.dtype, coords=self.coords,
                                     gids=self.gids, weights=self.weights,
                                     proc_weights=self.proc_weights,
                                     root=self.root)

        if self.lb_count:
            self.lb_obj.gather()

        # NOTE: Possible deadlock if some process doesn't enter this block
        if self.exec_count:
            self.lb_obj.adjust_proc_weights(self.exec_time, self.exec_nobjs,
                                        self.exec_count)

        self.lb_obj.load_balance()

        if self.lb_obj.rank == self.lb_obj.root and self.data:
            recvdtype = self.lb_obj.get_recvdtype(self.data[0].dtype)
            # FIXME: This is creating new arrays everytime
            aligned_data = carr.align(self.data, self.lb_obj.all_gids,
                                      backend=self.backend)
        else:
            aligned_data = []

        for i, senddata in enumerate(aligned_data):
            recvdata = carr.empty(self.lb_obj.plan.nreturn, dtype=recvdtype)
            self.lb_obj.comm_do(senddata, recvdata)
            setattr(self.lb_data, data_names[i], recvdata)

        for i, name in enumerate(self.coord_names):
            setattr(self.lb_data, '%s' % name, self.lb_obj.coords_view[i])

        setattr(self.lb_data, 'gids', self.lb_obj.gids_view)
        setattr(self.lb_data, 'weights', self.lb_obj.weights_view)
        setattr(self.lb_data, 'proc_weights', self.lb_obj.proc_weights_view)
        setattr(self.lb_data, 'num_objs', self.lb_obj.num_objs)

        self.exec_time = 0.
        self.exec_nobjs = 0
        self.exec_count = 0
        self.lb_count += 1
