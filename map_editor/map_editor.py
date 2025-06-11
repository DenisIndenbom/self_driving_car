import json

import keyboard as kb
import pygame

# Sorry for the crappy code. I wrote this code 2 years ago. I'm too lazy to rewrite it

pygame.init()
screen = pygame.display.set_mode((1280, 720))
screen.fill((0, 0, 0))
pygame.display.set_caption('map editor')
clock = pygame.time.Clock()

filename = 'map.json'
with open(filename, 'r') as f:
    data = json.loads(f.read())

env_map = data['borders'].copy()
rewards = data['rewards'].copy()
b_count = False

count = len(rewards) - 1

reward_mode = False
second_line_mode = False


def save_map():
    print('saving...')
    save = json.dumps({'borders': env_map, 'rewards': rewards})

    with open(filename, 'w') as file:
        file.write(save)

    print('save')


while True:
    screen.fill((0, 0, 0))

    for i in pygame.event.get():
        if i.type == pygame.QUIT:
            save_map()
            print('exit')
            exit()

        if i.type == pygame.MOUSEBUTTONDOWN:
            if i.button == 1:
                if not reward_mode:
                    if not second_line_mode:
                        env_map[0].append([i.pos[0], i.pos[1]])
                    else:
                        env_map[1].append([i.pos[0], i.pos[1]])

                    pygame.draw.circle(screen, (255, 0, 0), i.pos, 5)
                else:
                    if not b_count:
                        rewards.append([i.pos[0], i.pos[1]])
                        b_count = True
                    else:
                        rewards[count].append(i.pos[0])
                        rewards[count].append(i.pos[1])
                        count += 1
                        b_count = False

    if kb.is_pressed('q'):
        second_line_mode = True

    if kb.is_pressed('r'):
        reward_mode = True

    if kb.is_pressed('esc'):
        env_map = [[], []]
        rewards = []
        reward_mode = False
        second_line_mode = False
        b_count = False

    if kb.is_pressed('s'):
        save_map()

    if len(env_map) > 1:
        for row in env_map:
            for i in range(len(row)):
                try:
                    pygame.draw.line(screen, (255, 0, 0), (row[i][0], row[i][1]), (row[i + 1][0], row[i + 1][1]))
                except IndexError:
                    pass

        for row in rewards:
            try:
                pygame.draw.line(screen, (0, 255, 0), (row[0], row[1]), (row[2], row[3]))
            except Exception:
                pass

    pygame.display.update()
    clock.tick(20)
