mk_env_config(args) = Dict(
    "window" => args["window"],
    "grid" => (args["gx"], args["gy"]),
    "food_types" => args["food-types"],
    "latest_agent_ids" => [0, 0],
    "matchups" => [Tuple("f$(i)a0" for i in 0:args["rollout-group-size"]-1)],
    "episode_length" => args["episode-length"],
    "respawn" => true,
    "fires" => [Tuple(args["fires"][i:i+1]) for i in 1:2:length(args["fires"])],
    "foods" => [Tuple(args["foods"][i:i+2]) for i in 1:3:length(args["foods"])],
    "health_baseline" => true,
    "pickup_coeff" => args["pickup-coeff"],
    "light_coeff" => args["light-coeff"],
    "spawn_agents" => "center",
    "spawn_food" => "corner",
    "food_agent_start" => args["food-agent-start"],
    "food_env_spawn" => args["food-env-spawn"],
    "day_night_cycle" => true,
    "day_steps" => args["day-steps"],
    "seed" => args["seed"],
    "vocab_size" => 0)

update_df(df::Nothing, mets::Nothing) = nothing
update_df(df::Nothing, mets)   = DataFrame(mets)
update_df(df::DataFrame, mets) = push!(df, mets)
write_mets(file_name::String, df::Nothing) = nothing
write_mets(file_name::String, df::AbstractDataFrame) = CSV.write(file_name, df)
update_df_and_write_metrics(file_name::String, df::Nothing, mets::Nothing) = nothing
update_df_and_write_metrics(file_name::String, df, mets::Nothing) = nothing
function update_df_and_write_metrics(file_name::String, df, mets)
    df = update_df(df, mets)
    CSV.write(file_name, df)
    df
end

function average_walk(walks)
    """walks is ::Vector{Vector{Tuple{Float64, Float64}}}
    but too much of a pain to specify type in main script
    """
    avg_walk = []
    for step in zip(walks...)
        avg_step = mean.(zip(step...))
        push!(avg_walk, avg_step)
    end
    avg_walk
end

function average_bc(bcs::Vector)
  @assert Set(length.(bcs)) |> length == 1
  [mean(x) for x in zip(bcs...)]
end

function max_bc(bcs::Vector)
  @assert Set(length.(bcs)) |> length == 1
  [maximum(x) for x in zip(bcs...)]
end

function aggregate_metrics(metrics::Vector{<:AbstractDict})
  # we get a stack overflow error if we do all metrics in one call
  # so we do them one at a time
  agg_metrics = Dict()
  for met::AbstractDict in metrics
       mergewith!(vcat, agg_metrics, met)
  end
  agg_metrics
end

function aggregate_rollouts(fetches, pop_size)
    F = [[] for _ in 1:pop_size]
    BC = [[] for _ in 1:pop_size]
    walks_list = [[] for _ in 1:pop_size]
    group_rollout_metrics::Vector{Dict} = []
    for fet in fetches, idx in keys(fet[1])
        push!(F[idx], fet[1][idx]...)
        push!(BC[idx], fet[2][idx]...)
        push!(walks_list[idx], fet[3]["avg_walks"][idx]...)
        push!(group_rollout_metrics, fet[3]["mets"])
    end
    F = [mean(f) for f in F]
    BC = [average_bc(bcs) for bcs in BC]
    walks = [average_walk(w) for w in walks_list]
    rollout_metrics = aggregate_metrics(group_rollout_metrics)
    @assert length(F) == length(BC) == pop_size
    F, BC, walks, rollout_metrics
end


ts(x) = println(Dates.format(now(), "HH:MM:SS")*" $x")
ts(f, x) = println(f, Dates.format(now(), "HH:MM:SS")*" $x")

function log_mmm!(mets, name, arr)
    mets["$(name)_min"] = minimum(arr)
    mets["$(name)_mean"] = mean(arr)
    mets["$(name)_max"] = maximum(arr)
    mets["$(name)_std"] = std(arr)
end

function aid(i::Int, j::Int)
    if i > 0
        return "p$(i)_$j"
    elseif i < 0
        return "e$(-i)_$j"
    else
        throw("Invalid agent id $i")
    end
end

function invert(m::Dict)
    inverted_dict = Dict{valtype(m), Vector{keytype(m)}}()
    for (k, v) in m
        push!(get!(() -> valtype(inverted_dict)[], inverted_dict, v), k)
    end
    inverted_dict
end
