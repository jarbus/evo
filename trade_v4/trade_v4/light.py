import numpy as np
from typing import List, Tuple

FIRE_LIGHT_LEVEL:     float = 0.1
MAX_LIGHT_LEVEL:      float = 1.0
STARTING_LIGHT_LEVEL: float = 0.0

def isclose(a,b):
    return abs(a-b) <= 1e-09

class Light:
    def __init__(self, grid_size: tuple, fires: List[Tuple], interval):
        self.gx, self.gy = self.grid_size = grid_size
        self.light_level = 0
        self.interval = interval
        self.increasing = True
        self.fires = fires
        self.fire_radius = 3
        self.frame = self.fire_frame()

    def reset(self):
        self.light_level = STARTING_LIGHT_LEVEL
        self.increasing = True
        self.frame = self.fire_frame()

    def dawn(self):
        return self.increasing and isclose(self.light_level, 0)

    def contains(self, pos):
        return self.frame[pos] >= 0

    def fire_frame(self):
        if self.light_level >= 0:
            return np.full(self.grid_size, self.light_level)
        fire_light = np.zeros((self.gx, self.gy))
        for f in self.fires:
            for i in range(self.fire_radius):
                fire_light[max(0, f[0]-i):(f[0]+i+1), max(0, f[1]-i):(f[1]+i+1)] += 2*self.interval
        frame = np.full(self.grid_size, self.light_level)
        frame[fire_light > 0] = np.clip(fire_light[fire_light > 0] - (self.interval * 4), self.light_level, 1)
        return frame

    def step_light(self):
        if self.increasing:
            self.light_level += self.interval
        else:
            self.light_level -= self.interval
        if isclose(abs(self.light_level), MAX_LIGHT_LEVEL):
            self.increasing = not self.increasing
        self.frame = self.fire_frame()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    l = Light((11,11), [(1,1), (9,9)], 0.1)
    l.increasing = False
    for i in range(10):
        print(l.frame.round(2))
        plt.matshow(l.frame, vmin=-1, vmax=1)
        plt.colorbar()
        plt.show()
        plt.close()
        l.step_light()
