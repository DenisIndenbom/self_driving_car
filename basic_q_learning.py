from app import *

train = True

if __name__ == '__main__':
    app = App((config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT),
              save_path='agents/basic',
              genetic_algorithm=False,
              is_render=True)

    emap = Map('map.json')
    objs = [Car((520, 670), 0, emap) for _ in range(1)]
    agents = [QCarAgent(obj) for obj in objs]

    agents[0].load('agents/basic/agent_0.model')

    if not train:
        agents[0].eval()

    app.setup(emap, objs, agents)
    app.run(1000)

    exit()
