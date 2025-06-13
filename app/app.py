import os
import random

import pygame

from .base import Entity, Agent, Map, State

__all__ = ['App']


class App:
    def __init__(self, screen_size: tuple[int, int],
                 save_path: str = './',
                 genetic_algorithm: bool = False,
                 is_render: bool = True):
        self.screen_size = screen_size
        self.save_path = save_path
        self.genetic_algorithm = genetic_algorithm
        self.is_render = is_render

        if self.is_render:
            pygame.init()
            self.screen = pygame.display.set_mode(screen_size)
            pygame.display.set_caption('Self-Driving Car Simulator')
            self.clock = pygame.time.Clock()

        self.map = None
        self.objects: list[Entity] = []
        self.agents: list[Agent] = []

        self.states: list[State] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []

    def __del__(self):
        if self.is_render:
            pygame.quit()

    def setup(self, env_map: Map, objects: list[Entity], agents: list[Agent]):
        assert len(objects) == len(agents), 'Number of objects and agents must match.'
        self.map = env_map
        self.objects = objects
        self.agents = agents

    def run(self, epochs: int = 1, save_agents: bool = True):
        for epoch in range(epochs):
            self._reset_episode()
            delta_time = 1 / 60

            while not all(self.dones):
                if self._handle_events():
                    return

                self._step_agents(delta_time)

                if self.is_render:
                    delta_time = self._render()

            if self.genetic_algorithm:
                self._elect_best(0.25, 0.5, 0.05)  # Hard code, i know
            else:
                self._update_agents(save_agents)

            self._log_epoch(epoch)

    def _reset_episode(self):
        self.states = [obj.reset() for obj in self.objects]
        self.rewards = [0.0 for _ in self.objects]
        self.dones = [False for _ in self.objects]

    def _handle_events(self) -> bool:
        if self.is_render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return True

        return False

    def _step_agents(self, delta_time: float):
        for idx, (obj, agent, state) in enumerate(zip(self.objects, self.agents, self.states)):
            if self.dones[idx]:
                continue

            action = agent.step(state)
            new_state, reward, done = obj.update(action, delta_time)
            agent.observe(state, action, new_state, reward)

            self.states[idx] = new_state
            self.rewards[idx] += reward
            self.dones[idx] = done

    def _update_agents(self, save_agents: bool):
        for idx, agent in enumerate(self.agents):
            agent.update_policy()
            if save_agents:
                agent.save(os.path.join(self.save_path, f'agent_{idx}.model'))

    def _elect_best(self, election_size: float, fusion_ratio: float, mutation_rate: float):
        # Sort agents by reward (descending)
        ranked = sorted(zip(self.agents, self.rewards), key=lambda x: x[1], reverse=True)
        elite_cutoff = int(len(ranked) * election_size)
        top_agents = [agent for agent, _ in ranked[:elite_cutoff]]

        for idx, agent in enumerate(top_agents):
            agent.save(os.path.join(self.save_path, f'top_agent_{idx}.model'))

        # Generate new population
        new_agents = []
        while len(new_agents) < len(self.agents):
            parent1 = random.choice(top_agents)
            parent2 = random.choice(top_agents)
            child = parent1.merge_policy(parent2, fusion_ratio)
            child = child.mutate_policy(mutation_rate)
            new_agents.append(child)

        # Replace old agents
        self.agents = new_agents

    def _log_epoch(self, epoch: int):
        avg_reward = sum(self.rewards) / len(self.rewards)
        print(
            f'Epoch {epoch + 1} | Avg Reward: {avg_reward:.2f} | Max: {max(self.rewards):.2f} | Min: {min(self.rewards):.2f}'
        )

    def _render(self) -> float:
        self.screen.fill((0, 0, 0))

        self.map.draw(self.screen)

        for obj in self.objects:
            obj.render(self.screen)

        pygame.display.flip()

        return self.clock.tick(60) / 1000
