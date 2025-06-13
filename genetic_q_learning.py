from app import *

if __name__ == '__main__':
    app = App((config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT),
              save_path='agents/genetic',
              genetic_algorithm=True,
              is_render=True)

    emap = Map('map.json')
    objs = [Car((520, 670), 0, emap) for _ in range(50)]
    agents = [QCarAgent(obj) for obj in objs]

    app.setup(emap, objs, agents)
    app.run(1000)

    exit()
