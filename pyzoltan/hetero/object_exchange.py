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
