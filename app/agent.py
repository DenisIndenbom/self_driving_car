import pickle

import keyboard as kb
import numpy as np

from app.base import Agent, State, Action
from app.car import Car, CarAction

__all__ = ['PlayerAgent', 'QCarAgent']


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

    def observe(self, state: State, action: Action, new_state: State, reward: int | float):
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

    def save(self, path: str):
        pass

    def load(self, path: str):
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
        self.action_size = 6
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

    def observe(self, state: State, action: Action, new_state: State, reward: int | float):
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

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.policy = pickle.load(f)
