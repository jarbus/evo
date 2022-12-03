import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from .utils import inv_dist
from collections import defaultdict
class TradeCallback(DefaultCallbacks):

    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        env = base_env.get_unwrapped()[0]
        self.comm_history = [0 for i in range(env.vocab_size)]
        self.agent_dists = []
        self.action_counts = defaultdict(int, [(act, 0) for act in env.MOVES])
        self.punish_counts = defaultdict(int, [(agent, 0) for agent in env.agents])

    def on_episode_step(self, worker, base_env, policies, episode, **kwargs):
        # there is a bug where on_episode_step gets called where it shouldn't
        env = base_env.get_unwrapped()[0]
        for agent, comm in env.communications.items():
            if comm and max(comm) == 1:
                symbol = comm.index(1)
                self.comm_history[symbol] += 1
        dists = {}
        for i, a in enumerate(env.agents[:-1]):
            for j, b in enumerate(env.agents[i+1:]):
                if env.compute_done(a) or env.compute_done(b):
                    continue
                dists[(a,b)] = inv_dist(env.agent_positions[a], env.agent_positions[b])
        self.agent_dists.append(sum(dists.values()) / max(1, len(dists.values())))
        self.agent_grid = np.zeros(env.grid_size)
        for a in env.agents:
            if not env.compute_done(a):
                self.action_counts[env.MOVES[episode.last_action_for(a)]] += 1
                self.agent_grid[env.agent_positions[a]] += 1
        for aid, a in enumerate(env.agents):
            ax, ay = env.agent_positions[a]
            self.punish_counts[a] += np.sum(self.agent_grid * env.punish_frames[aid])\
                - env.punish_frames[aid, ax, ay]  # remove self-punish


    def on_episode_end(self, worker, base_env, policies, episode, **kwargs,):
        env = base_env.get_unwrapped()[0]
        #if not all(env.dones.values()):
        #    return
        #episode.custom_metrics["grid_size"] = env.grid_size
        for food, count in enumerate(env.mc.num_exchanges):
            episode.custom_metrics[f"exchange_{food}"] = count
        #for symbol, count in enumerate(self.comm_history):
        #    episode.custom_metrics[f"comm_{symbol}"] = count
        for agent in env.agents:
            #episode.custom_metrics[f"{agent}_punishes"] = self.punish_counts[agent]
            #episode.custom_metrics[f"{agent}_lifetime"] = env.mc.lifetimes[agent]
            episode.custom_metrics[f"{agent}_food_imbalance"] = \
                max(env.agent_food_counts[agent]) / max(1, min(env.agent_food_counts[agent]))
            total_agent_exchange = {"give": 0, "take": 0}
            for other_agent in env.agents:
                other_agent_exchange = {"give": 0, "take": 0}
                for food in range(env.food_types):
                    give = env.mc.player_exchanges[(agent, other_agent, food)]
                    take = env.mc.player_exchanges[(other_agent, agent, food)]
                    other_agent_exchange["give"] += give
                    other_agent_exchange["take"] += take
                    if other_agent != agent:
                        total_agent_exchange["give"] += give
                        total_agent_exchange["take"] += take
                #episode.custom_metrics[f"{agent}_take_from_{other_agent}"] = other_agent_exchange["take"]
                #episode.custom_metrics[f"{agent}_give_to_{other_agent}"] = other_agent_exchange["give"]
                episode.custom_metrics[f"{agent}_mut_exchange_{other_agent}"] =\
                   min(other_agent_exchange["take"],
                       other_agent_exchange["give"])
            #episode.custom_metrics[f"{agent}_take_from_all"] = total_agent_exchange["take"]
            #episode.custom_metrics[f"{agent}_give_to_all"] = total_agent_exchange["give"]
            episode.custom_metrics[f"{agent}_mut_exchange_total"] =\
                min(total_agent_exchange["take"], total_agent_exchange["give"])
            for food in range(env.food_types):
                episode.custom_metrics[f"{agent}_PICK_{food}"] = env.mc.picked_counts[agent][food]
                episode.custom_metrics[f"{agent}_PLACE_{food}"] = env.mc.placed_counts[agent][food]

        episode.custom_metrics["avg_avg_dist"] = sum(self.agent_dists) / len(self.agent_dists)
        total_number_of_actions = sum(self.action_counts.values())
        if total_number_of_actions > 0:
            for act, count in self.action_counts.items():
                episode.custom_metrics[f"percent_{act}"] = count / total_number_of_actions
        episode.custom_metrics["rew_base_health"] = env.mc.rew_base_health
        episode.custom_metrics["rew_light"] = env.mc.rew_light
        episode.custom_metrics["rew_act"] = env.mc.rew_acts
