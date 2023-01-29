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

function aggregate_metrics(batches::Vector{Batch})
  agg_metrics = Dict()
  for batch in batches
    mergewith!(vcat, agg_metrics, batch.mets)
  end
  agg_metrics
end
function aggregate_metrics(metrics::Vector{<:Dict})
  # we get a stack overflow error if we do all metrics in one call
  # so we do them one at a time
  agg_metrics = Dict()
  for met in metrics
       mergewith!(vcat, agg_metrics, met)
  end
  agg_metrics
end

function mk_id2idx_map(pop::Pop)
    id2idx = Dict{String, Int}()
    for (idx, ind) in enumerate(pop.inds)
        id2idx[ind.id] = idx
    end
    id2idx
end

function aggregate_rollouts!(rollouts::Vector{Batch}, pops::Vector{Pop})
  id2idx_maps = [mk_id2idx_map(pop) for pop in pops]
  for pop in pops, ind in pop.inds
    ind.fitnesses = []
    ind.bcs = []
    ind.walks = Vector{Walk}[]
  end

  group_rollout_metrics::Vector{Dict} = []
  for batch in rollouts
    push!(group_rollout_metrics, batch.mets)
    for id in keys(batch.rews)
      for (p, pop) in enumerate(pops)
        id ∉ keys(id2idx_maps[p]) && continue
        idx = id2idx_maps[p][id]
        push!(pop.inds[idx].fitnesses, batch.rews[id]...)
        push!(pop.inds[idx].bcs, batch.bcs[id]...)
        push!(pop.inds[idx].walks, batch.info["avg_walks"][id]...)
        break
      end
    end
  end
  for pop in pops
    for ind in pop.inds
      @assert length(ind.fitnesses) == length(ind.bcs) >= 1
      ind.fitnesses = [mean(ind.fitnesses)]
      ind.bcs = [average_bc(ind.bcs)]
      ind.walks = [average_walk(ind.walks)]
    end
  end
  rollout_metrics = aggregate_metrics(group_rollout_metrics)
  rollout_metrics
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

# for writing tests without having f0 everywhere
v32(x::Vector) = Vector{Float32}(x)

function invert(m::Dict)
  inverted_dict = Dict{valtype(m), Vector{keytype(m)}}()
  for (k, v) in m
    push!(get!(() -> valtype(inverted_dict)[], inverted_dict, v), k)
  end
  inverted_dict
end

aid(i::String, j::Int) = "$(i)_$j"

function mk_id_player_map(group::Vector{Ind})
  # if we have multiple of the same ind in a group,
  # we need to make sure that multiple players
  # are made
  counts = Dict{String, Int}()
  for ind in group
      counts[ind.id] = get(counts, ind.id, 0) + 1
  end
  id_map = Dict(aid(ind.id, c)=> ind.id 
            for ind in group
              for c in 1:counts[ind.id])
  id_map, counts
end

elite(x::Vector) = length(x) > 2 ? x[1:end-2] : x
