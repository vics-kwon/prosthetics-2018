from random_agent import RandomAgent
from osim.http.client import Client
import argparse


class RemoteSubmit(object):
    def __init__(self, token, agent_type):
        self.token = token
        self.remote_base = "http://grader.crowdai.org:1729"
        self.client = Client(self.remote_base)

        # TODO:: Add agent selector
        if agent_type == 'random':
            self.agent = RandomAgent()
        else:
            raise Exception('Not supported agent-type')

    def run(self):
        try:
            observation = self.client.env_create(self.token, env_id="ProstheticsEnv")

            while True:
                action = self.agent.get_action(observation)
                [observation, reward, done, info] = self.client.env_step(action, False)
                if done:
                    observation = self.client.env_reset()
                    if not observation:
                        break
            self.client.submit()
        except Exception as e:
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remote Submit to CrowdAI')
    parser.add_argument('-t', help='private token of CrowdAI', required=True, type=str, dest='token')
    parser.add_argument('-a', help='agent type', choices=['random'], required=True, type=str, dest='agent')

    args = parser.parse_args()
    token = args.token
    agent_type = args.agent

    try:
        submitter = RemoteSubmit(token=token, agent_type=agent_type)
        submitter.run()
    except Exception as e:
        print(e)
