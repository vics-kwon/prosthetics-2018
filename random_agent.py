from agent import AgentClass


class RandomAgent(AgentClass):
    def get_action(self, observation):
        return self.env.action_space.sample().tolist()
