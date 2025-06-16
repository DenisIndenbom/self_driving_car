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
                          9,
                          64,
                          lr=1e-3,
                          gamma=0.99,
                          target_update_freq=100,
                          epsilon_decay=0.955,
                          epsilon_min=0.05,
                          epsilon_max=1.0 if not load else 0.7,
                          batch_size=1024 * 16,
                          buffer_size=50000
                          )
              for obj in objs]

    if exists(join(save_path, 'agent_0000.model')) and load:
        agents[0].load(save_path)

    if not train:
        agents[0].eval()

    app.setup(emap, objs, agents)
    app.run(1000)

    exit()
