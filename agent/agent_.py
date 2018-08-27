from osim.env import ProstheticsEnv


class AgentClass(object):
    def __init__(self):
        self.env = ProstheticsEnv(visualize=False)

    def get_action(self, observation):
        pass
