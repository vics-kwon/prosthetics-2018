from agent import FixedActionAgent, RandomAgent, A3CAgent

import argparse
import json


class LocalTester(object):
    def __init__(self, agent_type):
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
            return status
        except Exception as e:
            status = {
                'status': 'ERROR',
                'error_msg': e
            }
            raise Exception(status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Local tester to CrowdAI')
    parser.add_argument('-a', help="agent type", choices=['random', 'fixed-action', 'a3c'], required=True, type=str, dest='agent')

    args = parser.parse_args()
    agent_type = args.agent

    try:
        tester = LocalTester(agent_type=agent_type)
        status = tester.run()
        print(json.dumps(status, indent=2))
    except Exception as e:
        print(json.dumps(e, indent=2))
