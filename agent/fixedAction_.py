'''
Thanks to Ryan :)
https://github.com/seungjaeryanlee/osim-rl-helper/blob/master/helper/baselines/FixedActionAgent.py
'''

from agent.agent_ import AgentClass


class FixedActionAgent(AgentClass):
    def get_action(self, observation):
        action = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0]
        return action
