from agent import FixedActionAgent, RandomAgent, A3CAgent

from osim.http.client import Client
import argparse
import json


class RemoteSubmit(object):
    def __init__(self, token, agent_type):
        self.token = token
        self.remote_base = "http://grader.crowdai.org:1729"
        self.client = Client(self.remote_base)

        # TODO:: Add agent selector
        if agent_type == 'random':
            self.agent = RandomAgent()
        elif agent_type == 'fixed-action':
            self.agent = FixedActionAgent()
        elif agent_type == 'a3c':
            self.agent = A3CAgent()
        else:
            status = {
                'status': 'ERROR',
                'error_msg': 'Not supported agent-type'
            }
            raise Exception(status)

    def run(self):
        try:
            status = self.agent.run()
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
            status = {
                'status': 'ERROR',
                'error_msg': e
            }
            raise Exception(status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remote Submit to CrowdAI')
    parser.add_argument('-t', help='private token of CrowdAI', required=True, type=str, dest='token')
    parser.add_argument('-a', help='agent type', choices=['random', 'fixed-action', 'a3c'], required=True, type=str, dest='agent')

    args = parser.parse_args()
    token = args.token
    agent_type = args.agent

    try:
        submitter = RemoteSubmit(token=token, agent_type=agent_type)
        submitter.run()
    except Exception as e:
        print(json.dumps(e, indent=2))
