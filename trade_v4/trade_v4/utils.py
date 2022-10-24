import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
from random import shuffle
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def add_tup(tup1: tuple, tup2: tuple):
    return tuple(map(lambda i, j: i + j, tup1, tup2))

def valid_pos(pos: tuple, grid_size: tuple):
    return all(0<=p<G for (p, G) in zip(pos, grid_size))

def inv_dist(tup1: tuple, tup2: tuple):
    return 1/(1+math.sqrt(sum((j-i)**2 for i,j in zip(tup1, tup2))))

def two_combos(xs: tuple, ys: tuple):
    return [(x, y) for x in xs for y in ys]

def punish_region(x, y, gx, gy):
    window = 1
    x_region = slice(max(0, x-window), min(gx, x+window))
    y_region = slice(max(0, y-window), min(gy, y+window))
    return x_region, y_region

def matchup_shuffler(matchups_list: List[Tuple]):
    matchups = matchups_list.copy()
    while True:
        shuffle(matchups)
        for matchup in matchups:
            yield matchup

#POLICY_MAPPING_FN = {
#    1: lambda aid, **kwargs: "pol1",
#    2: lambda aid, **kwargs: "pol1" if aid in {"player_0", "player_2"} else "pol2",
#    4: lambda aid, **kwargs: aid,
#    8: lambda aid, **kwargs: aid,
#}
def POLICY_MAPPING_FN(aid: str, *args ,**kwargs) -> str:
    return aid
