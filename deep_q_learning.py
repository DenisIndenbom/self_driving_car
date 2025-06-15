from os.path import exists, join

from app import *

train = True
load = True

save_path = 'agents/deep'

if __name__ == '__main__':
    app = App((config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT),
              save_path=save_path,
              genetic_algorithm=False,
              update_policy_freq=100,
              is_render=True)

    emap = Map('map.json')
    objs = [Car((520, 670), 0, emap) for _ in range(1)]
    agents = [DQNCarAgent(obj,
                          9,
                          6,
                          lr=1e-2,
                          gamma=0.99,
                          target_update_freq=100,
                          epsilon_decay=0.95,
                          epsilon_min=0.01,
                          epsilon_max=1.0 if not load else 0.01,
                          batch_size=512 * 8,
                          buffer_size=50000
                          )
              for obj in objs]

    if exists(join(save_path, 'agent_0.model')) and load:
        agents[0].load(join(save_path, 'agent_0.model'))

    if not train:
        agents[0].eval()

    app.setup(emap, objs, agents)
    app.run(1000)

    exit()
