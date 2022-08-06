import math

from pygame import image, Surface, transform

from abstract_classes import GameObject

from base_classes import *

from components import *

from ai import *

from utils import *

__all__ = ['Car']


class Car(GameObject):
    def __init__(self,
                 pos: tuple,
                 angle: float,
                 map: Map,
                 path: str = 'model.nn',
                 train: bool = True,
                 load_ai: bool = False):
        """
        This is class of car
        :param pos: car position 2d-tuple
        :param angle: car angle float
        :param map: game map class Map
        :param path: path to saved model
        """
        # base properties
        self.x, self.y = pos
        self.angle = angle
        self.map = map

        # spawn pos
        self.spawn_pos = (self.x, self.y)
        self.spawn_angle = self.angle

        # car sprite
        self.original_img = image.load("../sprites/Black_viper.png").convert_alpha()
        self.original_img = transform.scale(self.original_img, (int(214 * 0.2), int(100 * 0.2)))
        self.img = self.original_img

        # car properties
        self.speed = 0.
        self.max_speed = 6
        self.min_speed = -5
        self.angle_speed = 4

        self.gas_boost = 2
        self.brake_boost = 1

        self.friction = 0.02

        self.controllability = 1.2

        self.vec_angle = 0.

        # car collision
        self.collision = BoxCollision(self.img.get_width(), self.img.get_height(), self.map)

        # car AI
        self.ai = AIAgent(6, 7, 16, gpu_mode=True, model_path=path, load=load_ai, train=train)
        self.data = Data()
        self.train = train

    def update(self):
        self.img = rotate_img(self.original_img, self.angle)

        ladar_values = self.get_ladar_values()

        action_id = self.ai.predict(ladar_values)

        self.movement(action_id)

        flag = self.collision.update((self.x, self.y), self.angle)

        if flag == 0:
            self.respawn()

        if self.train:
            reward = self.get_reward(flag, ladar_values)

            self.data.add_row(action_id, ladar_values, reward)

            if flag == 0:
                self.ai.step(self.data)

    def render(self, screen: Surface):
        screen.blit(self.img, (self.x - self.img.get_width() / 2, self.y - self.img.get_height() / 2))

    def movement(self, action=-1):
        gas, brake, right, left = False, False, False, False

        if action == 0:
            gas = True
        elif action == 1:
            brake = True
        elif action == 2:
            gas = True
            right = True
        elif action == 3:
            gas = True
            left = True
        elif action == 4:
            brake = True
            right = True
        elif action == 5:
            brake = True
            left = True

        if gas:
            if self.speed < self.max_speed:
                self.speed += self.gas_boost
            if right:
                self.angle -= self.angle_speed
            if left:
                self.angle += self.angle_speed

        elif brake:
            if self.speed > self.min_speed:
                self.speed -= self.brake_boost
            if right:
                self.angle += self.angle_speed
            if left:
                self.angle -= self.angle_speed
        else:
            self.speed -= self.speed * self.friction

            if self.speed == 0:
                self.vec_angle = self.angle

        if self.vec_angle > self.angle:
            self.vec_angle -= self.controllability * self.angle_speed
        elif self.vec_angle < self.angle:
            self.vec_angle += self.controllability * self.angle_speed

        self.x += self.speed * math.cos(-self.vec_angle * math.pi / 180)
        self.y += self.speed * math.sin(-self.vec_angle * math.pi / 180)

    def get_ladar_values(self) -> list:
        start = (self.x, self.y)
        ladar_depth = 300

        ladar_values = []

        ladar_lines = \
            [
                Line(start, get_end_point_of_line(start, self.angle, 0, ladar_depth)),
                Line(start, get_end_point_of_line(start, self.angle, 30, ladar_depth)),
                Line(start, get_end_point_of_line(start, self.angle, -30, ladar_depth)),
                Line(start, get_end_point_of_line(start, self.angle, 80, ladar_depth)),
                Line(start, get_end_point_of_line(start, self.angle, -80, ladar_depth)),
                Line(start, get_end_point_of_line(start, self.angle, 140, ladar_depth)),
                Line(start, get_end_point_of_line(start, self.angle, -140, ladar_depth))
            ]

        for row in ladar_lines:
            minLength = 1
            distance = []
            for line in self.map.borders:
                success, x, y = line_intersection([row.start, row.end], [line.start, line.end])
                if success:
                    minLength = min(minLength, math.hypot(x - self.x, y - self.y) / ladar_depth)

                distance.append(minLength)

            ladar_values.append(min(distance))

        return ladar_values

    def respawn(self):
        # return car to spawn point
        self.x, self.y = self.spawn_pos

        self.speed = 0
        self.angle = self.spawn_angle
        self.vec_angle = self.angle

    @staticmethod
    def get_reward(flag: int, ladar_values: list) -> float:
        reward = 0

        if flag is None:
            reward += 0.2
        elif flag == 0:
            reward += -1.
        elif flag == 1:
            reward += 1.

        reward += sum(ladar_values) / len(ladar_values)

        return reward
