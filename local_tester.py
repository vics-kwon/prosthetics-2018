from random_agent import RandomAgent
from osim.env import ProstheticsEnv
import argparse


class LocalTester(object):
    def __init__(self, agent_type):
        # TODO:: Add agent selector
        if agent_type == 'random':
            self.agent = RandomAgent()
        else:
            raise Exception('Not supported agent-type')

    def run(self):
        env = ProstheticsEnv(visualize=False)
        observation = env.reset()
        total_reward = 0.0

        for i in range(200):
            action = self.agent.get_action(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break

        print("Total reward %f" % total_reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Local tester to CrowdAI')
    parser.add_argument('-a', help="agent type", choices=['random'], required=True, type=str, dest='agent')

    args = parser.parse_args()
    agent_type = args.agent

    try:
        tester = LocalTester(agent_type=agent_type)
        tester.run()
    except Exception as e:
        print(e)
