import math

from pygame import image, Surface, transform

from base import *
from components import *
from utils import *

__all__ = ['Car', 'CarState', 'CarAction']


class CarState(State):
    def __init__(self, ladars, speed, velocity):
        self.ladars = ladars
        self.speed = speed
        self.velocity = velocity

    def get(self):
        return self.ladars, self.speed, self.velocity


class CarAction(Action):
    def __init__(self, action_id):
        self.action_id = action_id

    def get(self):
        return self.action_id


class Car(Entity):
    def __init__(self,
                 pos: tuple,
                 angle: float,
                 env_map: Map):
        """
        This is class of car

        :param pos: position 2d-tuple
        :param angle: angle float
        :param env_map: world map Map
        """
        # base properties
        self.x, self.y = pos
        self.angle = angle
        self.map = env_map

        # spawn pos
        self.spawn_pos = (self.x, self.y)
        self.spawn_angle = self.angle

        # car sprite
        self.original_img = image.load('../sprites/Black_viper.png').convert_alpha()
        self.original_img = transform.scale(self.original_img, (int(214 * 0.2), int(100 * 0.2)))
        self.img = self.original_img

        # car properties
        self.speed = 0.
        self.max_speed = 10
        self.min_speed = -5
        self.angle_speed = 100

        self.gas_boost = 5
        self.brake_boost = 10

        self.friction = 0.5

        self.controllability = 1.2

        self.vec_angle = 0.

        # car collision
        self.collision = BoxCollision(self.img.get_width(), self.img.get_height(), self.map)

    def init_state(self) -> State:
        return CarState(self.get_ladar_values(), self.speed, self.angle)

    def update(self, action: Action, delta: float = 0.0) -> tuple[State, int | float, bool]:
        self.img = rotate_img(self.original_img, self.angle)

        self.movement(action.get(), delta)

        flag = self.collision.update((self.x, self.y), self.angle)

        done = False

        if flag == CollisionFlag.Collide:
            self.respawn()
            done = True

        ladars = self.get_ladar_values()

        return CarState(ladars, self.speed, self.angle), self.get_reward(flag), done

    def render(self, screen: Surface):
        screen.blit(self.img, (self.x - self.img.get_width() / 2, self.y - self.img.get_height() / 2))

    def movement(self, action=-1, delta: float = 0.0):
        gas, brake, right, left = False, False, False, False

        direction = action // 3
        turn = action % 3

        if direction == 1:
            gas = True
        elif direction == 2:
            brake = True

        if turn == 1:
            right = True
        elif turn == 2:
            left = True

        # Acceleration & braking
        if gas:
            effective_boost = self.gas_boost * (1 - (abs(self.speed) / self.max_speed) ** 2)
            if self.speed <= self.max_speed:
                self.speed += effective_boost * delta
        elif brake:
            effective_brake = self.brake_boost * (0.5 + 0.5 * (abs(self.speed) / self.max_speed))
            if self.speed >= self.min_speed:
                self.speed -= effective_brake * delta
        else:
            drag = (self.speed ** 2) * 0.001 * (1 if self.speed > 0 else -1)
            self.speed -= (drag + self.speed * self.friction) * delta

            if abs(self.speed) < 0.01:
                self.speed = 0
                self.vec_angle = self.angle

        # Steering
        steering_response = 0
        if right or left:
            # Increased turning responsiveness
            speed_factor = min(1.0, abs(self.speed) / (self.max_speed * 0.3))
            turn_direction = -1 if right else 1

            # Re-added delta scaling
            steering_response = turn_direction * self.angle_speed * speed_factor * delta

            if abs(self.speed) > self.max_speed * 0.7:
                slip_angle = steering_response * 0.4
                self.angle += slip_angle

        # Reduce steering under braking
        self.angle += steering_response * (1 if not brake else 0.7)

        # Align car's vector angle
        angle_diff = (self.angle - self.vec_angle + 180) % 360 - 180

        # Reducing speed on turns
        self.speed -= max(1.0, (abs(angle_diff) / 90) ** 2) * self.speed * self.friction * delta

        # Increased alignment speed
        alignment_speed = self.controllability * self.angle_speed * (
                0.4 + 0.6 * min(1.0, abs(self.speed) / self.max_speed)) * delta

        if angle_diff > 0:
            self.vec_angle += min(angle_diff, alignment_speed)
        elif angle_diff < 0:
            self.vec_angle -= min(-angle_diff, alignment_speed)

        # Position update
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
            min_length = 1
            distance = []
            for line in self.map.borders:
                success, x, y = line_intersection([row.start, row.end], [line.start, line.end])
                if success:
                    min_length = min(min_length, math.hypot(x - self.x, y - self.y) / ladar_depth)

                distance.append(min_length)

            ladar_values.append(min(distance) if len(distance) > 0 else 0)

        return ladar_values

    def respawn(self):
        # return car to spawn point
        self.x, self.y = self.spawn_pos

        self.speed = 0
        self.angle = self.spawn_angle
        self.vec_angle = self.angle

    @staticmethod
    def get_reward(flag: CollisionFlag) -> float:
        reward = 0.

        if flag == CollisionFlag.Collide:
            reward -= 1
        elif flag == CollisionFlag.Reward:
            reward += 0.5
        elif flag == CollisionFlag.Nothing:
            reward -= 0.1

        return reward
