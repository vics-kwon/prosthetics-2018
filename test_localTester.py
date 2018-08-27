from agent import FixedActionAgent, RandomAgent
from local_tester import LocalTester

from osim.env import ProstheticsEnv
from unittest import TestCase
import json


class TestLocalTester(TestCase):
    def test_LocalTester(self):
        agent_type1 = 'fixed-action'
        tester1 = LocalTester(agent_type=agent_type1)
        status1 = tester1.run()
        print(json.dumps(status1, indent=2))
        self.assertEqual(status1['status'], 'DONE')

        agent_type2 = 'stupid'
        with self.assertRaises(Exception):
            tester2 = LocalTester(agent_type=agent_type2)
            status2 = tester2.run()

    def test_RandomAgent(self):
        env = ProstheticsEnv(visualize=False)
        observation = env.reset()
        agent = RandomAgent()

        action = agent.get_action(observation)
        self.assertEqual(len(action), 19)

    def test_FixedActionAgent(self):
        env = ProstheticsEnv(visualize=False)
        observation = env.reset()
        agent = FixedActionAgent()

        action = agent.get_action(observation)
        self.assertEqual(len(action), 19)
