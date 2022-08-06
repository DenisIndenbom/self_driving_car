import json

from pygame import Surface, draw

__all__ = ['Line', 'Map']


class Line:
    def __init__(self, start=(0, 0), end=(0, 0)):
        self.start = start
        self.end = end


class Map:
    def __init__(self, path: str):
        self.borders = []
        self.rewards = []

        with open(path, 'r') as file:
            data = json.load(file)

        for row in data["borders"]:
            for i in range(len(row)):
                try:
                    self.borders.append(Line(row[i], row[i + 1]))
                except IndexError:
                    self.borders.append(Line(row[i], row[0]))

        for row in data["rewards"]:
            try:
                self.rewards.append(Line((int(row[0]), int(row[1])), (int(row[2]), int(row[3]))))
            except IndexError:
                pass

    def draw(self, screen: Surface):
        for row in self.borders:
            draw.line(screen, (100, 100, 100), row.start, row.end)

        for row in self.rewards:
            draw.line(screen, (0, 255, 0), row.start, row.end)
