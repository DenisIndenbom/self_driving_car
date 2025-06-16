import pickle
import random
from collections import deque
from os.path import join, exists

import keyboard as kb
import numpy as np
import torch
from torch import nn
from torch import optim

from app.base import Agent, State, Action
from app.car import Car, CarAction

__all__ = ['PlayerAgent', 'QCarAgent', 'DQNCarAgent']


class PlayerAgent(Agent):
    """ Default car agent for debugging """

    def step(self, state: State) -> Action:
        up = kb.is_pressed('up')
        down = kb.is_pressed('down')
        right = kb.is_pressed('right')
        left = kb.is_pressed('left')

        direction = 0
        if up and not down:
            direction = 1
        elif down and not up:
            direction = 2

        turn = 0
        if right and not left:
            turn = 1
        elif left and not right:
            turn = 2

        return CarAction((direction * 3) + turn)

    def observe(self, state: State, action: Action, new_state: State, reward: int | float, done: bool):
        pass

    def update_policy(self):
        pass

    def merge_policy(self, agent: Agent, ratio: float) -> Agent:
        pass

    def mutate_policy(self, mutation_rate: float) -> 'Agent':
        pass

    def eval(self):
        pass

    def train(self):
        pass

    def reset(self):
        pass

    def save(self, path: str, agent_id: int = 0):
        pass

    def load(self, path: str, agent_id: int = 0):
        pass


class QCarAgent(Agent):
    """ Car agent using Q-learning """

    def __init__(self,
                 car: Car,
                 discretization: int = 5,
                 learning_rate: float = 1e-3,
                 epsilon: float = 0.1,
                 gamma: float = 0.9):
        self.car = car
        self.discretization = discretization
        self.state_size = discretization ** car.ladar_num
        self.action_size = 9
        self.policy = np.random.rand(self.state_size, self.action_size)
        self.divisor = car.ladar_depth // discretization

        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.training = True

    def _discretize_ladars(self, ladars: list[float]) -> int:
        num_buckets = self.car.ladar_depth // self.divisor
        state_index = 0

        ladar_depth = self.car.ladar_depth

        for i, value in enumerate(ladars):
            bucket = min((value * ladar_depth) // self.divisor, num_buckets - 1)
            state_index = int(state_index * num_buckets + bucket)

        return state_index

    def step(self, state: State) -> Action:
        state_idx = self._discretize_ladars(state.get()[0])

        if self.training and np.random.random() < self.epsilon:
            action_idx = np.random.randint(self.action_size)
        else:
            action_idx = np.argmax(self.policy[state_idx])

        return CarAction(action_idx)

    def observe(self, state: State, action: Action, new_state: State, reward: int | float, done: bool):
        if not self.training:
            return

        # Handle input
        state_index = self._discretize_ladars(state.get()[0])
        action_index = action.get()
        new_state_index = self._discretize_ladars(new_state.get()[0])

        # Compute error
        best_next_action = np.max(self.policy[new_state_index])
        td_target = reward + self.gamma * best_next_action
        td_error = td_target - self.policy[state_index, action_index]

        # Update policy
        self.policy[state_index, action_index] += self.learning_rate * td_error

    def update_policy(self):
        pass

    def merge_policy(self, agent: Agent, ratio: float) -> Agent:
        if not isinstance(agent, QCarAgent):
            raise TypeError('Can only merge policies with another QCarAgent.')

        self.policy = (1 - ratio) * self.policy + ratio * agent.policy

        return self

    def mutate_policy(self, mutation_rate: float) -> 'Agent':
        mutation = np.random.rand(self.state_size, self.action_size)

        self.policy += mutation_rate * mutation

        return self

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def reset(self):
        self.policy = np.random.rand(self.state_size, self.action_size)

    def save(self, path: str, agent_id: int = 0):
        with open(join(path, f'agent_{agent_id}.model'), 'wb') as f:
            pickle.dump(self.policy, f)

    def load(self, path: str, agent_id: int = 0):
        with open(join(path, f'agent_{agent_id}.model'), 'rb') as f:
            self.policy = pickle.load(f)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DQN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.model(x)


class DQNCarAgent(Agent):
    """ Car agent using deep Q-learning """

    def __init__(self,
                 car: Car,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int,
                 lr: float = 1e-3,
                 gamma: float = 0.9,
                 target_update_freq: int = 1000,
                 epsilon_max: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.95,
                 batch_size=64,
                 buffer_size=10000):
        self.car = car
        self.training = True

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.lr = lr
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        self.updates_num = 0

        self.epsilon = epsilon_max
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.batch_size = batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.q_network = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.replay_buffer = deque(maxlen=buffer_size)

    def step(self, state: State) -> Action:
        if np.random.random() < self.epsilon and self.training:
            return CarAction(random.randint(0, self.action_dim - 1))

        ladars, speed, velocity = state.get()

        state = torch.FloatTensor(ladars + [speed, velocity]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)

        return CarAction(q_values.argmax().item())

    def observe(self, state: State, action: Action, new_state: State, reward: int | float, done):
        if not self.training:
            return

        ladars, speed, velocity = state.get()
        new_ladars, new_speed, new_velocity = new_state.get()

        processed_state = ladars + [speed / self.car.max_speed, velocity]
        processed_new_state = new_ladars + [new_speed / self.car.max_speed, new_velocity]

        processed_action = action.get()

        self.replay_buffer.append((processed_state, processed_action, reward, processed_new_state, done))

    def _sample_batch(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones

    def update_policy(self):
        if len(self.replay_buffer) < self.batch_size or not self.training:
            return

        states, actions, rewards, next_states, dones = self._sample_batch()

        current_q = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.updates_num += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.updates_num % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        print(f'[Update #{self.updates_num:04d}] '
              f'Loss: {loss.item():.4f} | '
              f'Epsilon: {self.epsilon:.4f} | '
              f'Buffer Size: {len(self.replay_buffer)}')

    def merge_policy(self, agent: Agent, ratio: float) -> Agent:
        raise NotImplementedError('DQN do not support merge policy')

    def mutate_policy(self, mutation_rate: float) -> 'Agent':
        raise NotImplementedError('DQN do not support mutate policy')

    def eval(self):
        self.training = False
        self.q_network.eval()

    def train(self):
        self.training = True
        self.q_network.train()

    def reset(self):
        raise NotImplementedError('DQN do not support reset policy')

    def save(self, path: str, agent_id: int = 0):
        torch.save(self.q_network.state_dict(), join(path, f'agent_{agent_id:04d}.model'))
        torch.save(self.optimizer.state_dict(), join(path, f'agent_{agent_id:04d}.optimizer'))

    def load(self, path: str, agent_id: int = 0):
        self.q_network.load_state_dict(torch.load(join(path, f'agent_{agent_id:04d}.model')))
        self.target_network.load_state_dict(self.q_network.state_dict())

        if exists(join(path, f'agent_{agent_id:04d}.optimizer')):
            self.optimizer.load_state_dict(torch.load(join(path, f'agent_{agent_id:04d}.optimizer')))
