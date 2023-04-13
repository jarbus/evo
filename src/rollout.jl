function run_batch(env::MazeEnv, models::Dict{String,<:Chain}, args; evaluation=false, render_str::Union{Nothing,String}=nothing, batch_size=nothing)

    @assert length(models) == 1
    key, model = first(models)
    batch_size = isnothing(batch_size) ? args["batch-size"] : batch_size
    @assert batch_size==1
    rewards, bcs = [], []
    #sample_act_func = evaluation ? x->[argmax(c) for c in eachcol(x)] : sample_batch
    sample_act_func = x->[argmax(c) for c in eachcol(x)]
    walk = Vector{Tuple{Float64,Float64}}()
    for i in 1:batch_size 
        reset!(env)
        r = -Inf
        for j in 1:args["episode-length"]
            obs = get_obs(env)
            probs = model(obs)
            acts = sample_act_func(probs)
            @assert length(acts) == 1
            r, done = Evo.Maze.step!(env, acts[1])
            push!(walk, (env.locations[4] .- 1))
            done && break
        end
        push!(rewards, r)
        push!(bcs, env.locations[4])
    end
    rews = Dict(key => mean(rewards))
    bc = Dict(key => average_bc(bcs))
    rews, Dict(), bc, Dict("avg_walks"=>Dict(key=>walk))
end

# Run trade
function run_batch(env_config::Dict, models::Dict{String,<:Chain}, args; evaluation=false, render_str::Union{Nothing,String}=nothing, batch_size=nothing)

    batch_size = isnothing(batch_size) ? args["batch-size"] : batch_size
    env_config["matchups"] = [tuple(keys(models)...)]
    b_env = [PyTrade().Trade(env_config) for _ in 1:batch_size]
    obs_size = (b_env[1].obs_size..., batch_size)
    num_actions = b_env[1].num_actions
    b_obs = batch_reset!(b_env, models)
    max_steps = args["episode-length"] * length(env_config["matchups"][1])
    rews = Dict(key => 0.0f0 for key in keys(models))
    avg_walks = Dict(key => Vector{NTuple{2,Float64}}() for key in keys(models))
    total_acts = Dict(key => Vector{Vector{UInt32}}() for key in keys(models))
    for _ in 1:max_steps
        b_obs, b_rew, b_dones, b_acts = batch_step!(b_env, models, b_obs, evaluation=evaluation)
        for (agent, avg_pos) in batch_pos!(b_env)
            push!(avg_walks[agent], avg_pos)
        end
        for name in keys(models)
            push!(total_acts[name], b_acts)
        end
        for (b, rew_dict) in enumerate(b_rew)
            for (name, rew) in rew_dict
                rews[name] += rew
                if render_str isa String && name == first(models).first
                    renderfile = "$render_str-$b.out"
                    # calls trade render for each step
                    try
                        render(b_env[b], renderfile)
                    catch
                        println("render failed for $renderfile")
                    end
                end
            end
        end
    end
    for (agent, walk) in avg_walks
        avg_walks[agent] = walk[1:4:end] # just return every 4th pos
    end
    rew_dict = Dict(name => rew / batch_size for (name, rew) in rews)
    mets = get_metrics(b_env)
    b_bc = [get_bcs(env) for env in b_env]
    bc = Dict(name => average_bc([bc[name] for bc in b_bc]) for name in keys(models))
    info = Dict{String, Any}("avg_walks"=>avg_walks)
    rew_dict, mets, bc, info
end
