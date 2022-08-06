from abc import ABC, abstractmethod
from pygame import Surface


__all__ = ['GameObject', 'Component']

class GameObject(ABC):
    x, y = 0, 0
    angle = 0.
    img: Surface = None

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def render(self, screen: Surface):
        pass


class Component(ABC):
    @abstractmethod
    def update(self, *args, **kwargs):
        pass