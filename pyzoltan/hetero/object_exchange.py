from pyzoltan.hetero.comm import dbg_print, Comm
import compyle.array as carr
import numpy as np


class ObjectExchange(object):
    def __init__(self):
        pass

    def set_plan(self, plan):
        self.plan = plan

    def comm_do_post(self, senddata, recvdata):
        self.plan.comm_do_post(senddata, recvdata)

    def comm_do_wait(self):
        self.plan.comm_do_wait()

    def comm_do(self, senddata, recvdata):
        self.plan.comm_do(senddata, recvdata)

    def invert(self):
        self.plan.invert_plan()

    def transfer(self):
        raise NotImplementedError("ObjectExchange::transfer called")

    def set_gather_plan(self):
        self.old_plan = self.plan
        proclist = carr.zeros(self.plan.nreturn, np.int32,
                              backend=self.plan.backend)
        self.plan = Comm(proclist, sorted=True, root=self.plan.root,
                         tag=self.plan.tag, backend=self.plan.backend)

    def gather(self):
        self.set_gather_plan()
        coords = self.transfer()
        self.plan = self.old_plan
        return coords
