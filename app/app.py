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

        self.map = Map('../empty_map.json')

        self.objects = [Car((520, 670), 180, self.map)]
        self.agents = [PlayerAgent()]
        self.states: list[State] = []

        assert len(self.objects) == len(self.agents)

    def run(self, epochs=1):
        for epoch in range(epochs):
            # Init states list
            self.states = []

            for obj in self.objects:
                self.states.append(obj.update(CarAction(0))[0])

            running = True
            delta = 1.0

            while running:
                if self.is_render and pygame.event.poll().type == pygame.QUIT:
                    pygame.quit()
                    break

                for obj, agent, (state_id, state) in zip(self.objects, self.agents, enumerate(self.states)):
                    # Predict the next action
                    action = agent.step(state)
                    new_state, reward, done = obj.update(action, delta)
                    agent.observe(state, action, new_state, reward)

                    # Update current states
                    self.states[state_id] = new_state

                    if done:
                        running = False
                        break

                if self.is_render:
                    delta = self.render()

    def render(self) -> float:
        self.screen.fill((0, 0, 0))

        self.map.draw(self.screen)

        for obj in self.objects:
            obj.render(self.screen)

        pygame.display.flip()

        return self.clock.tick(60) / 1000


if __name__ == '__main__':
    app = App(config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
    app.run()
