from pygame.transform import rotate
from pygame import Surface

import math

__all__ = ['rotate_img', 'line_intersection', 'get_end_point_of_line']


def rotate_img(img: Surface, angle: float) -> Surface:
    new_img = rotate(img, int(angle))

    return new_img


def get_end_point_of_line(start_pos: tuple, object_angle: int, line_angle: int, dis: int = 300) -> (float, float):
    x = start_pos[0] + dis * math.cos((-object_angle + line_angle) * math.pi / 180)
    y = start_pos[1] + dis * math.sin((-object_angle + line_angle) * math.pi / 180)

    return x, y


def line_intersection(line1: list, line2: list) -> (bool, float, float):
    k1 = (line1[1][0] - line1[0][0], line1[1][1] - line1[0][1])
    k2 = (line2[1][0] - line2[0][0], line2[1][1] - line2[0][1])

    rd = (line2[0][0] - line1[0][0], line2[0][1] - line1[0][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def in_range(n, d):
        if abs(d) < abs(n):
            return False
        if d > 0:
            return n >= 0
        else:
            return n <= 0

    div = det(k2, k1)
    n1 = det(k2, rd)
    if not in_range(n1, div):
        return False, 0, 0

    n2 = det(k1, rd)
    if not in_range(n2, div):
        return False, 0, 0

    t = n2 / div
    return True, line2[0][0] + t * k2[0], line2[0][1] + t * k2[1]
