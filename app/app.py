import pygame

import config
from agent import QCarAgent
from base import Entity, Agent, Map, State
from car import Car


class App:
    def __init__(self, screen_size: tuple[int, int], is_render=True):
        if is_render:
            pygame.init()
            self.screen = pygame.display.set_mode(screen_size)
            pygame.display.set_caption('self_driving_car')

            self.clock = pygame.time.Clock()

        self.is_render = is_render

        self.map = None

        self.objects = []
        self.agents = []

        self.states: list[State] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []

    def __del__(self):
        if self.is_render:
            pygame.quit()

    def setup(self, env_map: Map, objects: list[Entity], agents: list[Agent]):
        self.map = env_map

        self.objects = objects
        self.agents = agents

        assert len(self.objects) == len(self.agents)

    def run(self, epochs=1):
        for epoch in range(epochs):
            # Init states and reward list
            self.states = []
            self.rewards = []
            self.dones: list[bool] = [False] * len(self.agents)

            for obj in self.objects:
                self.states.append(obj.reset())
                self.rewards.append(0.0)

            delta = 1 / 60

            while True:
                if self.is_render and pygame.event.poll().type == pygame.QUIT:
                    pygame.quit()

                    return

                for idx, (obj, agent, state) in enumerate(zip(self.objects, self.agents, self.states)):
                    if self.dones[idx]:
                        continue

                    # Predict the next action
                    action = agent.step(state)
                    new_state, reward, done = obj.update(action, delta)
                    agent.observe(state, action, new_state, reward)

                    # Update current states
                    self.states[idx] = new_state
                    # Update total rewards
                    self.rewards[idx] += reward
                    # Update dones
                    self.dones[idx] = done

                if all(self.dones):
                    break

                if self.is_render:
                    delta = self.render()

            for idx, agent in enumerate(self.agents):
                agent.update_policy()
                agent.save(f'agent_{idx}.model')

            print(
                f'epoch: {epoch + 1}; avg reward: {sum(self.rewards) / len(self.rewards)}; max reward: {max(self.rewards)}; min reward: {min(self.rewards)}')

    def render(self) -> float:
        self.screen.fill((0, 0, 0))

        self.map.draw(self.screen)

        for obj in self.objects:
            obj.render(self.screen)

        pygame.display.flip()

        return self.clock.tick(60) / 1000


if __name__ == '__main__':
    app = App((config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT), is_render=False)

    emap = Map('../map.json')
    objs = [Car((520, 670), 0, emap)]
    agents = [QCarAgent(obj) for obj in objs]

    agents[0].load('agent.model')
    # agents[0].eval()

    app.setup(emap, objs, agents)
    app.run(10 ** 6)
    exit()
