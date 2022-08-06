import pygame
import config

import keyboard as kb

from base_classes import Map
from game_objects import Car

import keepAwake

keepAwake.enable()

class App:
    def __init__(self, display_width, display_height):
        pygame.init()
        self.screen = pygame.display.set_mode((display_width, display_height))
        self.screen.fill((135, 206, 250))
        pygame.display.set_caption("self-driving_car")

        self.clock = pygame.time.Clock()

        self.map = Map('../map.json')

        self.objects = [Car((520, 670), 0, self.map, load_ai=False)]

    @staticmethod
    def control():
        action_id = -1
        if kb.is_pressed("up"):
            action_id = 0
        if kb.is_pressed("down"):
            action_id = 1
        if kb.is_pressed("up") and kb.is_pressed("right"):
            action_id = 2
        if kb.is_pressed("up") and kb.is_pressed("left"):
            action_id = 3
        if kb.is_pressed("down") and kb.is_pressed("right"):
            action_id = 4
        if kb.is_pressed("down") and kb.is_pressed("left"):
            action_id = 5

        return action_id

    def run(self):
        while True:
            if pygame.event.poll().type == pygame.QUIT:
                exit()

            self.screen.fill((0, 0, 0))

            self.map.draw(self.screen)

            for obj in self.objects:
                obj.update()
                obj.render(self.screen)

            pygame.display.flip()
            self.clock.tick(60)

app = App(config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
app.run()