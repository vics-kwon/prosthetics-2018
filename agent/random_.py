from osim.env import ProstheticsEnv


class RandomAgent(object):
    def __init__(self):
        self.agent = AgentWorker()
        self.env = ProstheticsEnv(visualize=False)

    def run(self):
        try:
            observation = self.env.reset()
            total_reward = 0.0

            for i in range(200):
                action = self.agent.get_action(observation)
                observation, reward, done, info = self.env.step(action)
                total_reward += reward
                if done:
                    break
            print('Total reward %f' % total_reward)
            return {
                'status': 'DONE',
                'total reward': total_reward
            }
        except Exception as e:
            raise e

    def get_action(self, observation):
        action = self.agent.get_action(observation)
        return action


class AgentWorker(object):
    def __init__(self):
        self.env = ProstheticsEnv(visualize=False)

    def get_action(self, observation):
        return self.env.action_space.sample().tolist()
