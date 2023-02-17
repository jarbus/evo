#!/usr/bin/env python
import re
import gif
import argparse
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Tuple, List



#plt.style.use('dark_background')
parser = argparse.ArgumentParser(description="Convert serve file to gif")
parser.add_argument("file", type=str)
args = parser.parse_args()

player_expr = r"(.*): \((\d*), (\d*)\) \[[(\s*),\s]*(.*)\] (.*)$"
exchange_expr = r"Exchange: (.*) gave (\S*) of food (\d) to (.*)"
total_exchange_expr = r"Total exchanged.*: \[(.*), (.*)\]"
food_expr = r"food(\d):"
light_expr = r"Light:$"

@dataclass
class Player:
    name: str
    pos: Tuple[int, int]
    food_count: Tuple[float]
    done: bool

    def __str__(self):
        fcs = [f'{food:4.1f}' for food in self.food_count]
        return f"{self.name} {','.join(fcs)} {self.done}"

@dataclass
class Step:
    idx: int
    players: List[Player]
    exchange_messages: List[str]
    total_exchanged: List[float]
    food_grid: List[List[float]]
    light_grid: List[List[float]]


all_exchange_messages = []
player_colors = ["olivedrab", "darkcyan", "mediumslateblue", "purple", "brown", "navajowhite", "plum", "gray", "blue", "red", "orange"]
food_colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
offv = 0.015
player_offsets = (0, offv), (offv, 0), (0, 2*offv), (2*offv, 0),\
        (offv, offv), (offv, 2*offv), (2*offv, offv), (2*offv, 2*offv),\
        (0, offv), (offv, 0), (0, 2*offv), (2*offv, 0),\
        (offv, offv), (offv, 2*offv), (2*offv, offv), (2*offv, 2*offv)
grid_offset = (0.01, 0.01)
def add_tuple(a, b):
    return tuple(i + j for i, j in zip(a, b))
def mul_tuple(a: Tuple, b: float):
    return tuple(i * b for i in a)
def clip_tuple(min_: float, tup: Tuple, max_f: float):
    return tuple(min(max(min_, t), max_f) for t in tup)

@gif.frame
def plot_step(step: Step):

    fig = plt.figure()
    vs = 0.6
    hs = 0.6
    #axes.append(fig.add_axes([x, y, w, h]))
    grid = fig.add_axes([0.05, 0.15, vs - 0.1, 0.7])
    #grid.set_facecolor(f'{(float(step.light_level) + 1) / 2}')
    player_info = fig.add_axes([vs, hs, 1-vs, 1-hs])
    exchange_info = fig.add_axes([vs, 0, 1-vs, hs])
    scale = len(step.food_grid[0])

    for ax in [grid, player_info, exchange_info]:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.text(0, 0.95, f"Step {step.idx}", fontsize=12, wrap=True)
    for i, row in enumerate(step.light_grid[0]):
        for j, col in enumerate(row):
            l_pos = (j/scale, i/scale)
            # Add a square at l_pos
            not_yellow = (col + 1)/4
            yellow = (col + 1)/2
            rect = plt.Rectangle(l_pos, 1/scale, 1/scale, color=(yellow, yellow, not_yellow), fill=True)
            grid.add_patch(rect)

    fig.text(vs + 0.05, 1-0.05, "Player           reds      greens")
    fig.text(vs + 0.05, 1-0.07, "-----------------------------------------")
    fig.text(vs + 0.02, hs+0.03, f"Total Exchanged:", fontsize=8, wrap=True)
    for i, exchanged in enumerate(step.total_exchanged):
        fig.text(vs + 0.2 + (0.1 * i), hs+0.03, f"{exchanged}",
                    fontsize=10, wrap=True, family="monospace", color=mul_tuple(food_colors[i], 0.5))
    for i, message in enumerate(all_exchange_messages[-18:]):
        fig.text(vs + 0.02, hs-((i+1)*0.03), f"{message}", fontsize=8, wrap=True)
    for f, fg in enumerate(step.food_grid):
        for row, frow in enumerate(fg):
            for col, fcount in enumerate(frow):
                if fcount <= 0:
                    continue
                f_pos = (col/scale, row/scale)
                radius = 1/scale
                # Add base food color value with a scalar
                color = add_tuple(mul_tuple(food_colors[f], 0.2), mul_tuple(food_colors[f], fcount/4))
                color = clip_tuple(0.0, color, 1.0)
                circ = plt.Rectangle(f_pos, radius, radius, color=color, fill=True)
                grid.add_patch(circ)
    for i, player in enumerate(step.players):
        color = player_colors[i] if not player.done else "lightgrey"
        fig.text(vs + 0.05, 1-0.02-((i+2)*0.035), f"{player.name}", fontsize=10, wrap=True, family="monospace", color=color)
        for j, fc in enumerate(player.food_count):
            fig.text(vs + 0.2 + (.1 * j), 1-0.02-((i+2)*0.035), f"{round(fc, 1)}",
                    fontsize=10, wrap=True, family="monospace", color=mul_tuple(food_colors[j], 0.5))


        p_pos = tuple(p / scale for p in reversed(player.pos))

        p_pos = add_tuple(p_pos, (0, 0))
        p_pos = add_tuple(p_pos, (0.15/scale, 0.15/scale))
        p_pos = add_tuple(p_pos, player_offsets[i])
        radius = 0.5/scale
        # circ = plt.Circle(p_pos, radius=radius, color=color, fill=True)
        circ = plt.Rectangle(p_pos, radius, radius, color=color, fill=True)
        grid.add_patch(circ)
        if player.done:
            grid.text(*add_tuple(p_pos, (-radius/1.5, -radius/1.5)), f"{i}", fontsize=10, wrap=True, family="monospace")
        grid.add_patch(circ)



with open(args.file, "r") as file:
    lines = file.readlines()
    steps = []
    for i, line in enumerate(lines):
        if "STEP" in line or "game over" in line:
            steps.append(i)
step_slices = []
for i in range(len(steps)-1):
    start = steps[i] + 1
    end = steps[i+1]
    step_slices.append(slice(start, end))

frames = []

num_steps = len(step_slices)
#num_steps = 10  # len(step_slices)
for i in range(num_steps):
    step = Step(i, [],[], [], [], [])
    food = 0
    grid = step.food_grid
    for line in lines[step_slices[i]]:
        if m := re.match(player_expr, line):
            player, x, y, ft, done = m.groups()
            ft = ft.split(", ")
            p = Player(player, (int(x), int(y)), tuple(float(f) for f in ft), done == "True")
            step.players.append(p)

        if m := re.match(exchange_expr, line):
            giver, amount, food, taker = m.groups()
            step.exchange_messages.append(f"{giver} gave {amount} of {food} to {taker}")
            all_exchange_messages.append(f"{giver} gave {amount} of {food} to {taker}")

        if m := re.match(food_expr, line):
            food = m.groups()
            step.food_grid.append([])
            grid = step.food_grid

        if m := re.match(total_exchange_expr, line):
            total_exchanged = m.groups()
            step.total_exchanged = [float(f) for f in total_exchanged]

        if m := re.match(light_expr, line):
            # light = m.groups()[0]
            step.light_grid.append([])
            grid = step.light_grid
        # Food and Light
        if line.strip().startswith("["):
            grid[-1].append([float(f) for f in line.strip().replace("[", "").replace("]", "").replace(",","").split()])

    frame = plot_step(step)
    frames.append(frame)

# Specify the duration between frames (milliseconds) and save to file:
gif.save(frames, f'{args.file}.gif', duration=100)
