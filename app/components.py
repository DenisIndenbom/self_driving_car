import math
from enum import Enum

from .base import Map, Line, Component
from .utils import line_intersection

__all__ = ['BoxCollision', 'CollisionFlag']


class CollisionFlag(Enum):
    Collide = 0
    Reward = 1
    Nothing = 2


class BoxCollision(Component):
    def __init__(self, width, height, env_map: Map):
        self.width = width
        self.height = height

        self.lines = [Line() for _ in range(4)]

        self.map = env_map

    def update(self, pos, angle) -> CollisionFlag:
        """
        Box collision
        :param pos: object pos (tuple)
        :param angle: object angle (int)
        :return: will return 0 if object collide with a track, or will return 1 if object collide with a reward, or return None if object don't collide
        """
        angle_r = angle * math.pi / 180
        cos_a = math.cos(angle_r)
        sin_a = math.sin(angle_r)

        dwx = self.width / 2 * cos_a
        dwy = -self.width / 2 * sin_a

        dhx = self.height / 2 * sin_a
        dhy = self.height / 2 * cos_a

        fl = (pos[0] + dwx + dhx, pos[1] + dwy + dhy)
        fr = (pos[0] + dwx - dhx, pos[1] + dwy - dhy)
        rl = (pos[0] - dwx + dhx, pos[1] - dwy + dhy)
        rr = (pos[0] - dwx - dhx, pos[1] - dwy - dhy)

        self.lines[0] = Line(rl, fl)
        self.lines[1] = Line(fl, fr)
        self.lines[2] = Line(fr, rr)
        self.lines[3] = Line(rr, rl)

        for line in self.lines:
            for border in self.map.borders:
                success, x, y = line_intersection([line.start, line.end], [border.start, border.end])

                if success:
                    return CollisionFlag.Collide

        for line in self.lines:
            for reward in self.map.rewards:
                success, x, y = line_intersection([line.start, line.end], [reward.start, reward.end])

                if success:
                    return CollisionFlag.Reward

        # for row in self.lines:
        #     pygame.draw.line(self.scr,(255,0,0),row.start,row.end)

        return CollisionFlag.Nothing
