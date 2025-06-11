import pygame

import config
from agent import PlayerAgent
from base import Map, State
from car import Car, CarAction


class App:
    def __init__(self, display_width, display_height, is_render=True):
        if is_render:
            pygame.init()
            self.screen = pygame.display.set_mode((display_width, display_height))
            pygame.display.set_caption('self_driving_car')

            self.clock = pygame.time.Clock()

        self.is_render = is_render

        self.map = Map('../map.json')

        self.objects = [Car((520, 670), 180, self.map)]
        self.agents = [PlayerAgent()]
        self.states: list[State] = []

        assert len(self.objects) == len(self.agents)

    def run(self):
        # Init states list
        for obj in self.objects:
            self.states.append(obj.update(CarAction(0))[0])

        while True:
            if self.is_render and pygame.event.poll().type == pygame.QUIT:
                pygame.quit()
                break

            for obj, agent, (state_id, state) in zip(self.objects, self.agents, enumerate(self.states)):
                action = agent.step(state)
                new_state, reward = obj.update(action)
                agent.observe(state, action, new_state, reward)

                # Update current states
                self.states[state_id] = new_state

            if self.is_render:
                self.render()

    def render(self):
        self.screen.fill((0, 0, 0))

        self.map.draw(self.screen)

        for obj in self.objects:
            obj.render(self.screen)

        pygame.display.flip()
        self.clock.tick(60)


if __name__ == '__main__':
    app = App(config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
    app.run()
