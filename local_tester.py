from agent import FixedActionAgent, RandomAgent

from osim.env import ProstheticsEnv
import argparse
import json


class LocalTester(object):
    def __init__(self, agent_type):
        # TODO:: Add agent selector
        if agent_type == 'random':
            self.agent = RandomAgent()
        elif agent_type == 'fixed-action':
            self.agent = FixedActionAgent()
        else:
            status = {
                'status': 'ERROR',
                'error_msg': 'Not supported agent-type'
            }
            raise Exception(status)

    def run(self):
        try:
            env = ProstheticsEnv(visualize=False)
            observation = env.reset()
            total_reward = 0.0

            for i in range(200):
                action = self.agent.get_action(observation)
                observation, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
            # print('Total reward %f' % total_reward)
            return {
                'status': 'DONE',
                'total reward': total_reward
            }
        except Exception as e:
            status = {
                'status': 'ERROR',
                'error_msg': e
            }
            raise Exception(status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Local tester to CrowdAI')
    parser.add_argument('-a', help="agent type", choices=['random', 'fixed-action'], required=True, type=str, dest='agent')

    args = parser.parse_args()
    agent_type = args.agent

    try:
        tester = LocalTester(agent_type=agent_type)
        status = tester.run()
        print(json.dumps(status, indent=2))
    except Exception as e:
        print(json.dumps(e, indent=2))
