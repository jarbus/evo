import re
import os
import numpy as np
import random
from math import floor
from .utils import add_tup, directions, valid_pos, punish_region, matchup_shuffler, avg_tuple, get_strategy_name, STRATEGIES
from .light import Light
from .spawners import FireCornerSpawner, FoodSpawner, DiscreteFoodSpawner, CenterSpawner
import sys
from collections import defaultdict
from typing import List, Tuple, Dict

first_number_regex = re.compile(r'\d+')

def avg(x):
    return sum(x) / len(x)

def llogmmm(dic, name, arr):
    dic[f"{name}_min"] = min(arr)
    dic[f"{name}_max"] = max(arr)
    dic[f"{name}_mean"] = np.mean(arr)

class TradeMetricCollector():
    def __init__(self, env):
        self.poses = {agent: [] for agent in env.agents}
        self.num_exchanges = [0]*env.food_types
        self.picked_counts = {agent: [0] * env.food_types for agent in env.agents}
        self.placed_counts = {agent: [0] * env.food_types for agent in env.agents}
        self.player_exchanges = {(a, b, f): 0 for a in env.agents for b in env.agents for f in range(env.food_types)}
        self.lifetimes = {agent: 0 for agent in env.agents}
        # per agent dict for light reward
        self.night_penalties_avoided = {agent: 0 for agent in env.agents}
        self.rews = {agent: 0 for agent in env.agents}
        self.rew_health = {agent: 0 for agent in env.agents}
        self.rew_acts = {agent: 0 for agent in env.agents}
        self.rew_light = {agent: 0 for agent in env.agents}

    def collect_lifetimes(self, dones):
        for agent, done in dones.items():
            if not done:
                self.lifetimes[agent] += 1

    def collect_place(self, env, agent, food, actual_place_amount):
        self.placed_counts[agent][food] += actual_place_amount

    def collect_pick(self, env, agent, x, y, food, agent_id):
        exchange_amount = env.compute_exchange_amount(x, y, food, agent_id)
        if exchange_amount > 0:
            for i, other_agent in enumerate(env.agents):
                self.player_exchanges[(other_agent, agent, food)] += env.table[x, y, food, i]
        self.num_exchanges[food] += exchange_amount
        self.picked_counts[agent][food] += env.compute_pick_amount(x, y, food, agent_id)
        pass

    def collect_rews(self, agent, base_health, light, action_rewards):
        self.rew_health[agent] += base_health
        self.rew_light[agent]  += light
        self.rew_acts[agent]   += action_rewards

    def collect_pos(self, agent, pos):
        self.poses[agent].append(pos)

    def get_bcs(self, env):
        return {agent: self.get_bc(env, agent) for agent in env.agents}

    def get_bc8(self, env, agent):
        """
        BC is a length-8 list:
            0:1 picked counts
            2:3 placed counts
            4:5 avg pos
            6   light
            7   fitness
        """

        bc = [0 for _ in range(8)]
        f = 0 
        bc[0:env.food_types] = self.picked_counts[agent]
        f += env.food_types
        bc[f:f+env.food_types] = self.placed_counts[agent]
        f += env.food_types
        avg_pos = avg_tuple(self.poses[agent]) 
        avg_pos = [avg_pos[i] / env.grid_size[i] for i in range(len(env.grid_size))]
        bc[f:(f+2)] = avg_pos
        f +=2
        bc[f] = self.night_penalties_avoided[agent] / 3
        f +=1
        # we only want to compare positive rewards along this dimension, since we compare light penalty separately
        bc[f] = max(0, self.rews[agent])
        return bc

    def get_bc9(self, env, agent):
        """
        BC is a length-9 list:
            0:1 picked counts
            2:3 placed counts
            4:5 avg pos
            6   rew_light
            7   rew_base
            8   rew_acts
        """
        bc = [0 for _ in range(9)]
        f = 0 
        bc[0:env.food_types] = self.picked_counts[agent]
        f += env.food_types
        # an agent shouldn't place more than they forage
        # by repeatedly placing/picking
        clamped_placed = [min(pi, pl) for pi, pl in 
            zip(self.picked_counts[agent],
                self.placed_counts[agent])]
        bc[f:f+env.food_types] = clamped_placed
        f += env.food_types
        avg_pos = avg_tuple(self.poses[agent]) 
        avg_pos = [avg_pos[i] / env.grid_size[i] for i in range(len(env.grid_size))]
        bc[f:(f+2)] = avg_pos
        f +=2
        bc[f] = self.rew_light[agent]
        f +=1
        bc[f] = self.rew_health[agent]
        f +=1
        bc[f] = self.rew_acts[agent]
        # we only want to compare positive rewards along this dimension, since we compare light penalty separately
        return bc
    
    def get_bc(self, env, agent):
        """BC is a length-5 list:
            0:1 avg pos
            2   rew_light
            3   rew_health
            4   rew_light + rew_health
        """
        bc = [0 for _ in range(5)]
        f = 0 
        avg_pos = avg_tuple(self.poses[agent]) 
        avg_pos = [avg_pos[i] / env.grid_size[i] for i in range(len(env.grid_size))]
        bc[f:(f+2)] = avg_pos
        f +=2
        bc[f] = self.rew_light[agent]
        f +=1
        bc[f] = self.rew_health[agent]
        f +=1
        bc[f] = self.rew_light[agent] + self.rew_health[agent]
        return bc


    def return_metrics(self, env):
        custom_metrics = {
            "rew_base_health": [avg(self.rew_health.values())],
            "rew_light": [avg(self.rew_light.values())],
            "rew_acts": [avg(self.rew_acts.values())],
        }
        for st in STRATEGIES:
            custom_metrics[st] = []
        for food, count in enumerate(self.num_exchanges):
            custom_metrics[f"exchange_{food}"] = [count]
        #for symbol, count in enumerate(self.comm_history):
        #    episode.custom_metrics[f"comm_{symbol}"] = count
        food_imbalances = []
        takes = []
        gives = []
        picks = [[] for _ in range(env.food_types)]
        places = [[] for _ in range(env.food_types)]
        mut_exchanges = []

        for agent in env.agents:
            food_imbalances.append(
                max(env.agent_food_counts[agent]) / \
                max(1, min(env.agent_food_counts[agent])))
            total_agent_exchange = {"give": 0, "take": 0}
            for other_agent in env.agents:
                if other_agent == agent:
                    continue
                delta_foods = []
                for food in range(env.food_types):
                    give = self.player_exchanges[(agent, other_agent, food)]
                    take = self.player_exchanges[(other_agent, agent, food)]
                    total_agent_exchange["give"] += give
                    total_agent_exchange["take"] += take
                    delta_foods.append(give - take)

                # An exchange occured if there is a negative and positive delta
                if max(delta_foods) > 0 and min(delta_foods) < 0:
                    mut_exchanges.append(sum(delta_foods))
            
            gives.append(total_agent_exchange["give"])
            takes.append(total_agent_exchange["take"])
            strat = get_strategy_name(total_agent_exchange["give"], total_agent_exchange["take"])
            for st in STRATEGIES:
                custom_metrics[st].append(int(st == strat))
            for food in range(env.food_types):
                picks[food].append(self.picked_counts[agent][food])
                places[food].append(self.placed_counts[agent][food])
        mut_exchanges = mut_exchanges if len(mut_exchanges) > 0 else [0]

        custom_metrics["gives"] = gives
        custom_metrics["takes"] = takes
        custom_metrics["mut_exchanges"] = mut_exchanges
        custom_metrics["food_imbalances"] = food_imbalances
        for food in range(env.food_types):
            custom_metrics[f"picks_{food}"] = picks[food]
            custom_metrics[f"places_{food}"] = places[food]

        return custom_metrics




METABOLISM=0.1
PLACE_AMOUNT = 0.5
SCALE_DOWN = 5

NUM_ITERS = 100
ndir = len(directions)

class Trade:

    def __init__(self, env_config):
        #print(f"Creating Trade environment {env_config}")
        gx, gy = self.grid_size    = env_config.get("grid", (7, 7))
        self.food_types            = env_config.get("food_types", 2)
        #num_agents                 = env_config.get("num_agents", 2)
        self.matchups           = env_config.get("matchups")
        self.max_steps             = env_config.get("episode_length", 100)
        self.vocab_size            = env_config.get("vocab_size", 0)
        self.window_size           = env_config.get("window", (3, 3))
        self.dist_coeff            = env_config.get("dist_coeff", 0.0)
        self.move_coeff            = env_config.get("move_coeff", 0.0)
        self.death_prob            = env_config.get("death_prob", 0.1)
        self.day_night_cycle       = env_config.get("day_night_cycle", True)
        self.day_steps             = env_config.get("day_steps", 20)
        self.fires                 = env_config.get("fires")
        self.foods                 = env_config.get("foods")
        self.night_time_death_prob = env_config.get("night_time_death_prob", 0.1)
        self.punish                = env_config.get("punish", False)
        self.punish_coeff          = env_config.get("punish_coeff", 3)
        self.survival_bonus        = env_config.get("survival_bonus", 0.0)
        self.respawn               = env_config.get("respawn", True)
        self.light_coeff           = env_config.get("light_coeff", 1.0)
        self.pickup_coeff          = env_config.get("pickup_coeff", 1.0)
        self.health_baseline       = env_config.get("health_baseline", True)
        self.policy_mapping_fn     = env_config.get("policy_mapping_fn")
        self.food_env_spawn        = env_config.get("food_env_spawn")
        self.food_agent_start      = env_config.get("food_agent_start", 0)
        self.padded_grid_size      = add_tup(self.grid_size, add_tup(self.window_size, self.window_size))
        self.light                 = Light(self.grid_size, self.fires, 2/self.day_steps)
        if "seed" in env_config:
            self.seed(env_config["seed"])
        super().__init__()



        # old: extra info for trade
        # (self + policies) * (food frames and pos frame)
        #food_frame_and_agent_channels = (2) * (self.food_types+1)
        # x, y + agents_and_foods + food frames + comms
        #self.channels = 2 + food_frame_and_agent_channels + (self.food_types) + (self.vocab_size) + int(self.punish) + (2*int(self.day_night_cycle))

        # new: single food single agent forage only
        # food_frames and pos_frame
        food_frame_and_agent_channels = self.food_types+1
        # x + y + food_frames + pos_frame
        self.channels = 2 + food_frame_and_agent_channels
        self.agent_food_counts = dict()
        self.MOVES = ["UP", "DOWN", "LEFT", "RIGHT"]
        if self.punish:
            self.MOVES.append("PUNISH")
        for f in range(self.food_types):
            self.MOVES.extend([f"PICK_{f}", f"PLACE_{f}"])
        self.MOVES.extend([f"COMM_{c}" for c in range(self.vocab_size)])
        self.MOVES.append("NONE")
        self.num_actions = len(self.MOVES)
        self.communications = {}

        self.agent_spawner = CenterSpawner(self.grid_size)

        # Get rid of food type indicator for the spawner
        food_centers = [(fc[1], fc[2]) for fc in self.foods]
        self.food_spawner = FoodSpawner(self.grid_size, food_centers) 
        self.food_spawner = DiscreteFoodSpawner(self.grid_size, food_centers) 

        self.obs_size = (*add_tup(add_tup(self.window_size, self.window_size), (1, 1)), self.channels)
        self._skip_env_checking = True
        self.matchup_iterator = matchup_shuffler(self.matchups)

    def set_matchups(self, matchups: List[Tuple[str, str]]):
        self.matchups = matchups
        self.matchup_iterator = matchup_shuffler(self.matchups)
        self.reset()

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def generate_food(self):
        fc = self.food_env_spawn if self.respawn else 10
        #food_counts = [(0, fc), (0, fc), (1, fc), (1, fc)]
        food_counts = [(f[0], fc) for f in self.foods]
        self.table[:,:,:,:] = 0
        num_piles = int(fc)

        for i in range(num_piles):
            spawn_spots = self.food_spawner.gen_poses()
            for spawn_spot, (ft, fc) in zip(spawn_spots, food_counts):
                fx, fy = spawn_spot
                self.table[fx, fy, ft, len(self.agents)] += fc / num_piles

    def reset(self):


        if self.matchups == [] or self.matchups == [()]:
            return {}

        self.render_lines = []
        self.light.reset()
        self.agents = list(next(self.matchup_iterator)) # self.possible_agents[:]
        #print(f"Running env with agents {self.agents}")
        # print(f"resetting env with {self.agents}")
        self.action_rewards = {a: 0 for a in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.moved_last_turn = {agent: False for agent in self.agents}
        gx, gy = self.grid_size
        # use the last slot in the agents dimension to specify naturally spawning food
        self.table = np.zeros((*self.grid_size, self.food_types, len(self.agents)+1), dtype=np.float32)

        self.punish_frames = np.zeros((len(self.agents), *self.grid_size))
        self.generate_food()
        #self.agent_spawner.reset()
        spawn_spots = self.agent_spawner.gen_poses(len(self.agents))
        self.agent_positions = {agent: spawn_spot for agent, spawn_spot in zip(self.agents, spawn_spots)}
        self.steps = 0
        self.communications = {agent: [0 for j in range(self.vocab_size)] for agent in self.agents}
        self.agent_food_counts = {agent: [0 for f in range(self.food_types)] for agent in self.agents}
        # Each agent starts with self.food_agent_spawn of a single resource,
        # used for trading scenarios
        for i, a in enumerate(self.agents):
            self.agent_food_counts[a][i % self.food_types] = self.food_agent_start

        self.mc = TradeMetricCollector(self)
        return {self.agents[0]: self.compute_observation(self.agents[0])}

    def render(self, outfile=None):
        try:
            self.render_lines.append(f"--------STEP-{self.steps}------\n")
            for agent in self.agents:
                self.render_lines.append(f"{agent}: {self.agent_positions[agent]} {[round(fc, 2) for fc in self.agent_food_counts[agent]]} {self.compute_done(agent)}\n")
            for food in range(self.food_types):
                self.render_lines.append(f"food{food}:\n")
                for row in self.table[:,:,food].sum(axis=2).round(2):
                    self.render_lines.append(str(list(row)).replace(",","")+"\n")
            self.render_lines.append(f"Total exchanged so far: {self.mc.num_exchanges}\n")
            if self.day_night_cycle:
                self.render_lines.append(f"Light:\n")
                for row in self.light.frame.round(2):
                    self.render_lines.append(str(list(row)).replace(",","")+"\n")
            for agent, comm in self.communications.items():
                if comm and max(comm) >= 1:
                    self.render_lines.append(f"{agent} said {comm.index(1)}\n")
            if all(self.dones.values()) or self.steps >= self.max_steps:
                out= open(outfile, "a") if outfile else sys.stdout
                out.write("".join(self.render_lines))
                if outfile:
                    out.close()
        except Exception as e:
            print(e)

    def compute_observation2(self, agent):
        ax, ay = self.agent_positions[agent]
        wx, wy = self.window_size
        gx, gy = self.grid_size

        minx, maxx = ax, ax+(2*wx)+1
        miny, maxy = ay, ay+(2*wy)+1
        food_frames = self.table.sum(axis=3).transpose(2, 0, 1)  # frame for each food
        comm_frames = np.zeros((self.vocab_size, *self.grid_size), dtype=np.float32)
        self_pos_frame   = np.zeros(self.grid_size, dtype=np.float32)
        self_pos_frame[ax, ay] = 1
        self_food_frames = np.zeros((self.food_types, *self.grid_size), dtype=np.float32)
        pol_pos_frames   = np.zeros(self.grid_size, dtype=np.float32)
        pol_food_frames  = np.zeros((self.food_types, *self.grid_size), dtype=np.float32)
        for i, a in enumerate(self.agents):
            if self.compute_done(a):
                continue
            oax, oay = self.agent_positions[a]
            comm_frames[:, oax, oay] = self.communications[a]

            if a != agent:
                pol_pos_frames[oax, oay] += 1
                pol_food_frames[:, oax, oay] += self.agent_food_counts[a]
            else:
                self_food_frames[:,  oax, oay] += self.agent_food_counts[a]

        pol_frames = np.stack([*pol_food_frames, pol_pos_frames])
        agent_and_food_frames = np.stack([*pol_frames, self_pos_frame, *self_food_frames])

        if self.punish:
            pun_frames = np.sum(self.punish_frames, axis=0)[None, :, :]
        else:
            pun_frames = np.zeros((0, *self.grid_size), dtype=np.float32)

        if self.day_night_cycle:
            light_frames = self.light.frame[None, :, :] * SCALE_DOWN
        else:
            light_frames = np.zeros((0, *self.grid_size), dtype=np.float32)


        xpos_frame = np.repeat(np.arange(gy).reshape(1, gy), gx, axis=0) / gx
        ypos_frame = np.repeat(np.arange(gx).reshape(gx, 1), gy, axis=1) / gy

        frames = np.stack([*food_frames, *agent_and_food_frames, xpos_frame, ypos_frame, *light_frames, *pun_frames, *comm_frames], axis=2)
        padded_frames = np.full((*self.padded_grid_size, frames.shape[2]), -1, dtype=np.float32)
        padded_frames[wx:(gx+wx), wy:(gy+wy), :] = frames
        obs = padded_frames[minx:maxx, miny:maxy, :] / SCALE_DOWN
        return obs[:,:,:,None]

    def compute_observation(self, agent):
        ax, ay = self.agent_positions[agent]
        wx, wy = self.window_size
        gx, gy = self.grid_size

        minx, maxx = ax, ax+(2*wx)+1
        miny, maxy = ay, ay+(2*wy)+1
        full_frames = np.zeros((*self.grid_size, self.channels), dtype=np.float32)
        # Table
        fnum = 0
        full_frames[:,:, fnum:self.food_types] = self.table.sum(axis=3) / SCALE_DOWN  # frame for each food
        fnum += self.food_types
        # Self position
        full_frames[ax,ay, fnum] = 1
        fnum +=1
        ## Self food
        #full_frames[ax, ay, fnum:fnum+self.food_types] += self.agent_food_counts[agent]
        #full_frames[ax, ay, fnum:fnum+self.food_types] /= SCALE_DOWN
        #fnum += self.food_types
        ## Others pos, Others food, comms
        #pol_pos_frames = full_frames[:,:,fnum]
        #fnum += 1
        #pol_food_frames = full_frames[:,:,fnum:fnum+self.food_types]
        #fnum += self.food_types
        #comm_frames = full_frames[:,:,fnum:fnum+self.vocab_size]
        #fnum += self.vocab_size

        #for i, a in enumerate(self.agents):
        #    if self.compute_done(a):
        #        continue
        #    oax, oay = self.agent_positions[a]
        #    comm_frames[oax, oay, :] = self.communications[a]
        #    if a != agent:
        #        pol_pos_frames[oax, oay] += 1
        #        pol_food_frames[oax, oay, :] += self.agent_food_counts[a]
        #pol_food_frames /= SCALE_DOWN
        ## Light
        #if self.day_night_cycle:
        #    full_frames[:,:,fnum] = self.light.frame[:, :]
        #    fnum += 1
        #    full_frames[:,:,fnum] = self.light.campfire_frame[:, :]
        #    fnum += 1
        ## punish ?
        #if self.punish:
        #    full_frames[:,:,fnum] = np.sum(self.punish_frames, axis=0)[None, :, :]
        #    fnum += 1

        full_frames[:,:,fnum] = np.repeat(np.arange(gy).reshape(1, gy), gx, axis=0) / gy
        fnum+=1
        full_frames[:,:,fnum] = np.repeat(np.arange(gx).reshape(gx, 1), gy, axis=1) / gx
        fnum+=1

        padded_frames = np.full((*self.padded_grid_size, full_frames.shape[2]), -1, dtype=np.float32)
        padded_frames[wx:(gx+wx), wy:(gy+wy), :] = full_frames
        obs = padded_frames[minx:maxx, miny:maxy, :]
        return obs[:,:,:,None]

    def compute_done(self, agent):
        if self.dones[agent] or self.steps >= self.max_steps:
            return True
        return False


    def compute_reward(self, agent):
        # reward for each living player
        rew = 0
        if self.compute_done(agent):
            return rew

        num_of_food_types = sum(1 for f in self.agent_food_counts[agent] if f >= 0.1)
        base_health = [0, 0.1, 1][num_of_food_types]
        if self.light.contains(self.agent_positions[agent]):
            light_rew = 0 
        else:
            pos = self.agent_positions[agent]
            light_rew = self.light_coeff *\
                self.light.frame[pos]

        act_rew = self.pickup_coeff * self.action_rewards[agent]

        # Remember to update this function whenever you add a new reward
        self.mc.collect_rews(agent, base_health, light_rew, act_rew)

        rew  = base_health + light_rew + act_rew
        return rew

    def compute_exchange_amount(self, x: int, y: int, food: int, picker: int):
        return sum(count for a, count in enumerate(self.table[x][y][food]) if a != picker and a != len(self.agents))

    def compute_pick_amount(self, x: int, y: int, food: int, picker: int):
        return self.table[x][y][food][len(self.agents)] * self.collection_modifier(self.agents[picker], food)

    def update_dones(self):
        for agent in self.agents:
            if self.compute_done(agent):
                continue
            self.dones[agent] = self.steps >= self.max_steps
    def next_agent(self, agent):
        return self.agents[(self.agents.index(agent)+1) % len(self.agents)]

    def collection_modifier(self, agent: str, food: int):
        # search the string for the first number
        match = first_number_regex.search(agent)
        if match is None:
            raise ValueError(f"Agent name {agent} does not contain a number")
        pop_idx = int(match.group()) - 1 # convert from 1,2 to 0,1
        if pop_idx not in range(0, self.food_types):
            raise ValueError(f"Agent name {agent} is from population that is outside food types")

        if pop_idx == food:
            return 1.0
        return 0.5


    def step(self, actions):
        # placed goods will not be available until next turn
        for agent in actions.keys():
            if agent not in self.agents:
                breakpoint()
                print(f"ERROR: received act for {agent} which is not in {self.agents}")
        for agent, action in actions.items():
            self.action_rewards[agent] = 0
            # MOVEMENT
            self.moved_last_turn[agent] = False
            x, y = self.agent_positions[agent]
            self.mc.collect_pos(agent, (x,y))
            aid: int = self.agents.index(agent)
            if action in range(0, ndir):
                new_pos = add_tup(self.agent_positions[agent], directions[action])
                if valid_pos(new_pos, self.grid_size):
                    self.agent_positions[agent] = new_pos
                    self.moved_last_turn[agent] = True
            # punish
            elif action in range(ndir, ndir + int(self.punish)):
                x_pun_region, y_pun_region = punish_region(x, y, *self.grid_size)
                self.punish_frames[aid, x_pun_region, y_pun_region] = 1

            elif action in range(ndir + int(self.punish), ndir + int(self.punish) + (self.food_types * 2)):
                pick = ((action - ndir - int(self.punish)) % 2 == 0)
                food = floor((action - ndir - int(self.punish)) / 2)
                if pick:
                    self.mc.collect_pick(self, agent, x, y, food, aid)
                    self.agent_food_counts[agent][food] += np.sum(self.table[x, y, food,:-1])+\
                        self.table[x,y,food,-1]*self.collection_modifier(agent, food)
                    self.action_rewards[agent] += np.sum(self.table[x, y, food, -1])*\
                        self.collection_modifier(agent, food)
                    self.table[x, y, food, :] = 0
                elif self.agent_food_counts[agent][food] >= PLACE_AMOUNT:
                    actual_place_amount = PLACE_AMOUNT
                    self.agent_food_counts[agent][food] -= actual_place_amount
                    self.table[x, y, food, aid] += actual_place_amount
                    self.mc.collect_place(self, agent, food, actual_place_amount)
            # last action is noop
            elif action in range(4 + self.food_types * 2 + int(self.punish), self.num_actions-1):
                symbol = action - (self.food_types * 2) - 4
                assert symbol in range(self.vocab_size)
                self.communications[agent][symbol] = 1
            self.agent_food_counts[agent] = [max(x - METABOLISM, 0) for x in self.agent_food_counts[agent]]

            if agent == self.agents[-1]:
                self.steps += 1
                self.light.step_light()
                # Once agents complete all actions, add placed food to table
                if self.respawn and self.light.dawn():
                    self.generate_food()

                    self.update_dones()

        obs = {self.next_agent(agent): self.compute_observation(self.next_agent(agent)) for agent in actions.keys()}
        dones = {agent: self.compute_done(agent) for agent in actions.keys()}
        # self.mc.collect_lifetimes(dones)
        rewards = {self.next_agent(agent): self.compute_reward(self.next_agent(agent)) for agent in actions.keys()}

        dones = {**dones, "__all__": all(dones.values())}
        return obs, rewards, dones
