#This code is from RL-Adventure-2
#https://github.com/higgsfield/RL-Adventure-2/blob/master/1.actor-critic.ipynb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from osim.env import ProstheticsEnv

from utils.multiprocessing_env import SubprocVecEnv


class A3CAgent(object):
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.env = ProstheticsEnv(visualize=False)

        self.num_envs = 2

        self.envs = [self.make_env() for i in range(self.num_envs)]
        self.envs = SubprocVecEnv(self.envs)

        # num_inputs = self.envs.observation_space.shape[0]  # 158
        num_inputs = 160
        num_outputs = self.envs.action_space.shape[0]  # 19
        hidden_size = 256

        self.model = A3CWorker(num_inputs, num_outputs, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())

    def run(self):
        self.__run()
        total_reward = self.test_env()
        return {
            'status': 'DONE',
            'total reward': total_reward
        }

    def __run(self):
        lr = 3e-4
        num_steps = 10

        max_frames = 50000
        frame_idx = 0
        test_rewards = []

        state = self.envs.reset()
        while frame_idx < max_frames:
            log_probs = []
            values = []
            rewards = []
            masks = []
            entropy = 0

            for _ in range(num_steps):
                state = torch.FloatTensor(state).to(self.device)
                action, value = self.model(state)
                if self.device == 'cuda':
                    action = action.cpu().numpy()
                if self.device == 'cpu':
                    action = action.data.numpy()
                next_state, reward, done, _ = self.envs.step(action)

                log_prob = F.log_softmax(action, dim=1)
                entropy += log_prob.mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(self.device))

                state = next_state
                frame_idx += 1

                if frame_idx % 1000 == 0:
                    test_rewards.append(np.mean([self.test_env() for _ in range(10)]))
                    # plot(frame_idx, test_rewards)

            next_state = torch.FloatTensor(next_state).to(self.device)
            _, next_value = self.model(next_state)
            returns = self.compute_returns(next_value, rewards, masks)

            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns).detach()
            values = torch.cat(values)

            advantage = returns - values

            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
            print('frame_idx: ', frame_idx, 'loss: ', loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def test_env(self, vis=False):
        state = self.env.reset()
        if vis: self.env.render()
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).to(self.device)
            action, _ = self.model(state)

            if self.device == 'cuda':
                action = action.cpu().numpy()
            if self.device == 'cpu':
                action = action.data.numpy()

            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            if vis: self.env.render()
            total_reward += reward
        return total_reward

    def get_action(self, observation):
        state = torch.FloatTensor(observation).to(self.device)
        action, _ = self.model(state)
        return action.data.numpy()

    @staticmethod
    def make_env():
        def _thunk():
            env = ProstheticsEnv(visualize=False)
            return env
        return _thunk

    @staticmethod
    def compute_returns(next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns


class A3CWorker(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(A3CWorker, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs)
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        return probs, value
