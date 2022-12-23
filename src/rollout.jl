function run_batch(env::MazeEnv, models::Dict{String,<:Chain}, args; evaluation=false, render_str::Union{Nothing,String}=nothing)

    batch_size = args["batch-size"]
    EvoTrade.Maze.reset!(env)
    rewards, bcs = [], []
    sample_act_func = evaluation ? x->[argmax(c) for c in eachcol(x)] : sample_batch
    for i in 1:batch_size 
        r = -Inf
        for i in 1:args["episode-length"]
            obs = get_obs(env)
            probs = models["f0a0"](obs)
            acts = sample_act_func(probs)
            @assert length(acts) == 1
            r, done = EvoTrade.Maze.step!(env, acts[1])
            done && break
        end
        push!(rewards, r)
        push!(bcs, env.locations[4])
    end
    rews = Dict("f0a0" => mean(rewards), "f1a0"=> mean(rewards))
    bc = Dict("f0a0" => average_bc(bcs), "f1a0"=> average_bc(bcs))
    rews, nothing, bc, nothing
end

# Run trade
function run_batch(env_config::Dict, models::Dict{String,<:Chain}, args; evaluation=false, render_str::Union{Nothing,String}=nothing)

    batch_size = args["batch-size"]
    b_env = [PyTrade().Trade(env_config) for _ in 1:batch_size]
    obs_size = (b_env[1].obs_size..., batch_size)
    num_actions = b_env[1].num_actions
    b_obs = batch_reset!(b_env, models)
    max_steps = args["episode-length"] * args["num-agents"]
    rews = Dict(key => 0.0f0 for key in keys(models))
    avg_walks = Dict(key => Vector{NTuple{2,Float64}}() for key in keys(models))
    total_acts = Dict(key => Vector{Vector{UInt32}}() for key in keys(models))
    for _ in 1:max_steps
        b_obs, b_rew, b_dones, b_acts = batch_step!(b_env, models, b_obs, evaluation=evaluation)
        for (agent, avg_pos) in batch_pos!(b_env)
            push!(avg_walks[agent], avg_pos)
        end
        for (b, rew_dict) in enumerate(b_rew)
            for (name, rew) in rew_dict
                push!(total_acts[name], b_acts)
                rews[name] += rew
                if render_str isa String && name == first(models).first
                    renderfile = "$render_str/b$b.out"
                    # calls trade render for each step
                    render(b_env[b], renderfile)
                end
            end
        end
    end
    rew_dict = Dict(name => rew / batch_size for (name, rew) in rews)
    mets = get_metrics(b_env)
    bc = Dict(name => bc2(total_acts[name], num_actions) for (name, _) in models)
    info = Dict("avg_walks"=>avg_walks)
    rew_dict, mets, bc, info
end
