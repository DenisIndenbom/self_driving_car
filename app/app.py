import keyboard as kb
import pygame

import config
from base import Map
from car import Car, CarAction


class App:
    def __init__(self, display_width, display_height):
        pygame.init()
        self.screen = pygame.display.set_mode((display_width, display_height))
        self.screen.fill((135, 206, 250))
        pygame.display.set_caption('self_driving_car')

        self.clock = pygame.time.Clock()

        self.map = Map('../map.json')

        self.objects = [Car((520, 670), 0, self.map)]

    @staticmethod
    def control():
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

    def run(self):
        while True:
            if pygame.event.poll().type == pygame.QUIT:
                exit()

            self.screen.fill((0, 0, 0))

            self.map.draw(self.screen)

            for obj in self.objects:
                obj.update(self.control())
                obj.render(self.screen)

            pygame.display.flip()
            self.clock.tick(60)


if __name__ == '__main__':
    app = App(config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
    app.run()
