from trade_v4 import Trade
env_config = {
    "grid": (7, 7),
    "food_types": 1,
    "num_agents": 1,
    "matchups": [("f0a0",)],
    "episode_length": 100,
    "vocab_size": 0,
    "window": (3, 3),
    "day_night_cycle": True,
    "day_steps": 20,
    "fires": [(1,1)],
    "foods": [(0, 2,3)],
    "light_coeff": 1.0,
    "pickup_coeff": 1.0,
    "health_baseline": True,
    "policy_mapping_fn": None,
    "food_env_spawn": 1,
    "food_agent_start": 0
}

env = Trade(env_config)
env.reset()
obs, rews, dones = env.step({"f0a0": 0})
for i in range(obs["f0a0"].shape[2]):
    frame = obs["f0a0"][:,:,i]
    print(frame[:,:,0].round(2))
