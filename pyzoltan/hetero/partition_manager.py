from pyzoltan.load_balancer import LoadBalancer
from pyzoltan.cell_map import CellMap


class PartitionManager(object):
    def __init__(self, ndims, dtype, root=0, backend=None):
        if not mpi.Is_initialized():
            mpi.Init()
        self.comm = mpi.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.root = root
        self.backend = get_backend(backend)

        self.ndims = ndims
        self.dtype = dtype
        self.exec_time = 0.
        self.exec_nobjs = 0
        self.exec_count = 0
        self.lb_count = 0
        self.iter_count = 0

        self.weights = None
        self.gids = None
        self.proc_weights = None

        self.cell_map = CellMap(self.backend)

    def set_lbfreq(self, lbfreq):
        self.lbfreq = lbfreq

    def set_gids(self, gids):
        self.gids = gids

    def set_coords(self, coords):
        self.coords = coords

    def set_weights(self, weights):
        self.weights = weights

    def set_proc_weights(self, proc_weights):
        self.proc_weights = proc_weights

    def setup_cell_manager(self, cell_manager):
        self.cell_manager = cell_manager

    def set_object_exchange(self, object_exchange):
        self.object_exchange = object_exchange

    def setup_load_balancer(self):
        self.load_balancer = LoadBalancer(
                self.ndims, self.dtype, self.cell_manager,
                proc_weights=self.proc_weights, root=self.root,
                tag=self.tag, backend=self.backend
                )
        self.load_balancer.set_cell_map(self.cell_map)

    def update(self, migrate=True):
        if self.iter_count % self.lbfreq == 0:
            # gather everything
            self.cell_manager.generate_cells()
            self.comm.Bcast(self.cell_manager.ncells_per_dim.get_buff(),
                            root=self.root)
            plan = self.load_balance()
            # make ghost plan
        elif migrate:
            plan = self.migrate_objects()
        self.object_exchange.set_plan(plan)
        # call plan in object exchange
        self.iter_count += 1

    def migrate_objects(self):
        pass

    def load_balance(self):
        # there are 3 types of comm plans. The plan with first time
        # load balance, the plan with subsequent load balances
        # and the plan with migrate objects
        # All properties have to be sent for each of the plans.
        # The plan should be owned by the object exchange instance
        pass
