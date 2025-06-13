import json
import sys
from copy import deepcopy

import pygame

# Constants
WIDTH, HEIGHT = 1280, 720
FILENAME = 'map.json'
FPS = 60
LINE_COLOR = (255, 0, 0)
REWARD_COLOR = (0, 255, 0)
GRID_COLOR = (40, 40, 40)
BG_COLOR = (0, 0, 0)
POINT_RADIUS = 3
GRID_SIZE = 20

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('map editor')
clock = pygame.time.Clock()


# Load map data
def load_map(filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data.get('borders', [[]]), data.get('rewards', [])
    except FileNotFoundError:
        return [[]], []


# Save map data
def save_map(borders, rewards):
    with open(FILENAME, 'w') as f:
        json.dump({'borders': borders, 'rewards': rewards}, f)
    print("Saved.")


# Draw helpers
def draw_grid(offset, zoom):
    for x in range(0, WIDTH, GRID_SIZE):
        pygame.draw.line(screen, GRID_COLOR, ((x + offset[0]) * zoom, 0), ((x + offset[0]) * zoom, HEIGHT))
    for y in range(0, HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (0, (y + offset[1]) * zoom), (WIDTH, (y + offset[1]) * zoom))


def draw_lines(points, color, zoom, offset):
    for i in range(len(points) - 1):
        start = [(points[i][0] + offset[0]) * zoom, (points[i][1] + offset[1]) * zoom]
        end = [(points[i + 1][0] + offset[0]) * zoom, (points[i + 1][1] + offset[1]) * zoom]
        pygame.draw.line(screen, color, start, end, 2)


def draw_rewards(rewards, zoom, offset):
    for reward in rewards:
        if len(reward) == 4:
            start = [(reward[0] + offset[0]) * zoom, (reward[1] + offset[1]) * zoom]
            end = [(reward[2] + offset[0]) * zoom, (reward[3] + offset[1]) * zoom]
            pygame.draw.line(screen, REWARD_COLOR, start, end, 2)


def draw_points(points, color, zoom, offset):
    for point in points:
        end = [(point[0] + offset[0]) * zoom, (point[1] + offset[1]) * zoom]
        pygame.draw.circle(screen, color, end, POINT_RADIUS)


# Snap point to grid
def snap_to_grid(pos):
    return [round(pos[0] / GRID_SIZE) * GRID_SIZE, round(pos[1] / GRID_SIZE) * GRID_SIZE]


def main():
    borders, rewards = load_map(FILENAME)
    layer_index = 0
    reward_mode = False
    placing_reward_point = None
    undo_stack = []
    offset = [0, 0]
    zoom = 1.0
    pan_speed = 10

    running = True
    while running:
        screen.fill(BG_COLOR)
        draw_grid(offset, zoom)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_map(borders, rewards)
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = (event.pos[0] / zoom - offset[0], event.pos[1] / zoom - offset[1])
                pos = snap_to_grid([x, y])

                undo_stack.append((deepcopy(borders), deepcopy(rewards)))

                if reward_mode:
                    if placing_reward_point is None:
                        placing_reward_point = pos
                    else:
                        rewards.append([*placing_reward_point, *pos])
                        placing_reward_point = None
                else:
                    while len(borders) <= layer_index:
                        borders.append([])

                    borders[layer_index].append(pos)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    if undo_stack:
                        borders, rewards = undo_stack.pop()

                if event.key == pygame.K_s:
                    save_map(borders, rewards)
                if event.key == pygame.K_ESCAPE:
                    borders, rewards = [[]], []
                    placing_reward_point = None
                    undo_stack = []
                if event.key == pygame.K_r:
                    reward_mode = not reward_mode
                if event.key == pygame.K_TAB:
                    layer_index = (layer_index + 1) % max(len(borders), 1)
                    print(f"Switched to layer {layer_index}")
                if event.key == pygame.K_EQUALS:
                    zoom = min(5, zoom + 0.1)
                if event.key == pygame.K_MINUS:
                    zoom = max(0.2, zoom - 0.1)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            offset[0] += pan_speed / zoom
        if keys[pygame.K_RIGHT]:
            offset[0] -= pan_speed / zoom
        if keys[pygame.K_UP]:
            offset[1] += pan_speed / zoom
        if keys[pygame.K_DOWN]:
            offset[1] -= pan_speed / zoom

        for idx, line in enumerate(borders):
            color = LINE_COLOR if idx == layer_index else (100, 0, 0)
            draw_lines(line, color, zoom, offset)
            draw_points(line, color, zoom, offset)

        draw_rewards(rewards, zoom, offset)
        if placing_reward_point is not None:
            draw_points([placing_reward_point], REWARD_COLOR, zoom, offset)

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == '__main__':
    main()
