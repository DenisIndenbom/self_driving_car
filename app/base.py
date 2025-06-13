import json
from abc import ABC, abstractmethod

import pygame
from pygame import Surface, draw

__all__ = ['Line', 'Map', 'Entity', 'Component', 'Agent', 'State', 'Action']


class Line:
    def __init__(self, start=(0, 0), end=(0, 0)):
        self.start = start
        self.end = end

    def draw(self, screen):
        pygame.draw.line(screen, (0, 255, 0), self.start, self.end, 3)


class Map:
    def __init__(self, path: str):
        self.borders = []
        self.rewards = []

        with open(path, 'r') as file:
            data = json.load(file)

        for row in data['borders']:
            for i in range(len(row)):
                try:
                    self.borders.append(Line(row[i], row[i + 1]))
                except IndexError:
                    self.borders.append(Line(row[i], row[0]))

        for row in data['rewards']:
            try:
                self.rewards.append(Line((int(row[0]), int(row[1])), (int(row[2]), int(row[3]))))
            except IndexError:
                pass  # TODO: Fix this crunch

    def draw(self, screen: Surface):
        for row in self.borders:
            draw.line(screen, (100, 100, 100), row.start, row.end)

        for row in self.rewards:
            draw.line(screen, (0, 255, 0), row.start, row.end)


class State(ABC):
    @abstractmethod
    def get(self):
        pass


class Action(ABC):
    @abstractmethod
    def get(self):
        pass


class Entity(ABC):
    @abstractmethod
    def reset(self) -> State:
        pass

    @abstractmethod
    def update(self, action: Action, delta: float = 0.0) -> tuple[State, int | float, bool]:
        pass

    @abstractmethod
    def render(self, screen: Surface):
        pass


class Component(ABC):
    @abstractmethod
    def update(self, *args, **kwargs):
        pass


class Agent(ABC):
    @abstractmethod
    def step(self, state: State) -> Action:
        pass

    @abstractmethod
    def observe(self, state: State, action: Action, new_state: State, reward: int | float):
        pass

    @abstractmethod
    def update_policy(self):
        pass

    @abstractmethod
    def merge_policy(self, agent: 'Agent', ratio: float) -> 'Agent':
        """
        Merge policy of another agent and return a new agent.

        :param agent: Agent to merge.
        :param ratio: Ratio of new agent to merge.
        :return: New agent.
        """
        pass

    @abstractmethod
    def mutate_policy(self, mutation_rate: float) -> 'Agent':
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass
