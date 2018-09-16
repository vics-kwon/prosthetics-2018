from local_tester import LocalTester

from unittest import TestCase
import json


class TestLocalTester(TestCase):
    def test_LocalTester(self):
        agent_type = 'stupid'
        with self.assertRaises(Exception):
            tester = LocalTester(agent_type=agent_type)
            status = tester.run()

    def test_RandomAgent(self):
        agent_type = 'random'
        tester = LocalTester(agent_type=agent_type)
        status = tester.run()
        print(json.dumps(status, indent=2))
        self.assertEqual(status['status'], 'DONE')

    def test_FixedActionAgent(self):
        agent_type = 'fixed-action'
        tester = LocalTester(agent_type=agent_type)
        status = tester.run()
        print(json.dumps(status, indent=2))
        self.assertEqual(status['status'], 'DONE')

    def test_A3CAgent(self):
        agent_type = 'a3c'
        tester = LocalTester(agent_type=agent_type)
        status = tester.run()
        print(json.dumps(status, indent=2))
        self.assertEqual(status['status'], 'DONE')

    def test_A2CAgent(self):
        agent_type = 'a2c'
        tester = LocalTester(agent_type=agent_type)
        status = tester.run()
        print(json.dumps(status, indent=2))
        self.assertEqual(status['status'], 'DONE')

