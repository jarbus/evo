using CSV
using Distributed
using DataFrames
using FileIO
using Infiltrator
using Logging

@everywhere begin
  using EvoTrade
  args = $args
  expname = args["exp-name"]
  clsname = args["cls-name"]
  using Flux
  using Statistics
  using StableRNGs

  mets_to_return = "gives takes exchange_0 
  picks_0 places_0 exchange_1 picks_1 places_1
  strat_noop strat_give strat_take strat_exchange
  rew_base_health rew_acts rew_light mut_exchanges" |> split

  env_config = mk_env_config(args)

  function fitness(group::Vector, eval_gen)
    # count up any duplicate members
    counts = Dict{Int, Int}()
    for (i, _) in group
        counts[i] = get(counts, i, 0) + 1
    end
    global prefixes
    group = decompress_group(group, prefixes)
    # assign a player name like p[idx]_[count]
    models = Dict(aid(i, c)=>re(reconstruct(sc, mi, seeds, e_idxs))
              for (i, seeds, e_idxs) in group
              for c in 1:counts[i])
    rew_dict, mets, bc_dict, info_dict = run_batch(env, models, args, evaluation=true)
    rews, bcs, infos = Dict(), Dict(), Dict{Any, Any}("avg_walks"=>Dict())
    # convert from player name mapping back to index mapping
    for (i, _) in group 
      i < 0 && continue # skip elites
      aids = [aid(i, c) for c in 1:counts[i]]
      rews[i] = [rew_dict[aid_] for aid_ in aids]
      bcs[i] = [bc_dict[aid_] for aid_ in aids]
      infos["avg_walks"][i] = eval_gen ? [info_dict["avg_walks"][aid_] for aid_ in aids] : [[]]
    end
    infos["mets"] = eval_gen ? filter(p->p.first in mets_to_return, mets) : Dict()
    rews, bcs, infos
  end
end

function main()
  dt_str = args["datime"]
  logname="runs/$clsname/$dt_str-$expname.log"
  check_name = "outs/$clsname/$expname/check.jld2"
  met_csv_name = "outs/$clsname/$expname/metrics.csv"
  start_gen = 1
  global_logger(EvoTradeLogger(args["local"] ? stdout : logname))
  df = nothing
  wp = WorkerPool(workers())
  pop_size = args["pop-size"]
  pop = [Vector{Float32}([rand(UInt32)]) for _ in 1:pop_size]
  elites = Vector{Dict}()
  best = (-Inf, [])
  archive = Set()
  BC = nothing
  F = nothing
  γ = args["exploration-rate"]

  @info "cls: $clsname"
  @info "exp: $expname"
  @info "Running on commit: "*read(`git rev-parse --short HEAD`, String)
  
  @info "Initializing on all workers"
  @everywhere begin
    if !isnothing(args["maze"])
        env = maze_from_file(args["maze"])
        grid = env.grid 
        group_fn = singleton_groups
    else
        env = PyTrade().Trade(env_config)
        EvoTrade.Trade.reset!(env)
        grid = env.table 
        group_fn = one_v_self#all_v_all
    end

    m = make_model(env, args)
    θ, re = Flux.destructure(m)
    mi = ModelInfo(m, re)
    model_size = length(θ)
    # pass mazeenv struct or trade config dict
    env = env isa MazeEnv ? env : env_config
    global sc = SeedCache(maxsize=args["num-elites"]*3)
    prefixes = Dict()
  end
  @info "model has $model_size params"

  if isfile(check_name)
    @info "Loading from checkpoints"
    df = isfile(met_csv_name) ? CSV.read(met_csv_name, DataFrame) : nothing 
    global check = try 
        load(check_name)
    catch 
        load(check_name*".backup") 
    end
    start_gen = check["gen"] + 1
    γ = check["gamma"]
    F, BC, best, archive, novelties = getindex.((check,), ["F", "BC", "best","archive", "novelties"])
    global pop = check["pop"]
    global elites = check["elites"]
    global sc = check["sc"]
    cache_elites!(sc, mi, elites)
    @info "resuming from gen $start_gen"
  end

  for g in start_gen:args["num-gens"]
    global prefixes
    @info "starting generation $g"
    eval_gen = g % 2 == 1
    @info "compressing pop"
    rollout_pop = compress_pop(pop, elites, prefixes)
    @info "creating rollout groups"
    groups = group_fn(rollout_pop)
    @info "pmapping"
    fetches = pmap(wp, groups) do g
        fitness(g, eval_gen)
    end
    
    F, BC, walks, rollout_metrics = 
        aggregate_rollouts(fetches, pop_size)

    if maximum(F) > best[1]
        γ = 0.0
        best = (maximum(F), pop[argmax(F)], BC[argmax(F)])
        @info "New best ind found, F=$(best[1]), γ decreased to $γ"
    else
        # TODO change gamma to clamped value once GA test passes
        #γ = clamp(γ + 0.02, 0, 0.9)
        γ = 0.0
        @info "no better elite found, set γ to $γ"
    end
    
    add_to_archive!(archive, BC, pop, args["archive-prob"])
    bc_matrix = hcat(BC...)
    pop_and_arch = hcat([bc for (bc, _) in archive]..., bc_matrix)

    @info "computing novelties"
    novelties = compute_novelties(bc_matrix, pop_and_arch, k=min(pop_size-1, 25))
    @info "most novel bc: $(BC[argmax(novelties)])"
    @info "most fit bc: $(BC[argmax(F)]), fitness $(maximum(F))"

    if eval_gen # collect data only on evaluation generations
      @info "log start"
      metrics_csv = Dict()
      outdir = "outs/$clsname/$expname/"*string(g, pad=3, base=10)
      run(`mkdir -p $outdir`)

      @info "Running elite eval"
      n_fit_elites = ceil(Int, args["num-elites"]*(1-γ))
      eval_members = rollout_pop[sortperm(F, rev=true)[1:n_fit_elites]]
      eval_groups = group_fn(eval_members)
      eval_metrics = pmap(wp, eval_groups) do group
         group = decompress_group(group, prefixes)
         models = Dict("p$i-$c" => 
              re(reconstruct(sc, mi, seeds, eidxs))
              for (c, (i, seeds, eidxs)) in enumerate(group))
         str_name = joinpath(outdir, string(hash(group))*"-"*string(myid()))
         metrics= run_batch(env, models, args, evaluation=true,
                            render_str=str_name)[2]
         metrics
      end |> aggregate_metrics
      @info "Logging metrics"
      global rollout_metrics
      for (met_name, met_vec) in rollout_metrics
          log_mmm!(metrics_csv, "pop_"*met_name, met_vec)
      end
      for (met_name, met_vec) in eval_metrics
          log_mmm!(metrics_csv, "eval_"*met_name, met_vec)
      end
      log_mmm!(metrics_csv, "fitness", F)
      log_mmm!(metrics_csv, "novelty", novelties)
      metrics_csv["gamma"] = γ
      df = update_df_and_write_metrics(met_csv_name, df, metrics_csv)

      @info "Visualizing outs"
      isnothing(args["maze"]) && vis_outs(outdir, args["local"])
      plot_grid_and_walks(env, "$outdir/pop.png", grid, walks,
                          novelties, F, args["num-elites"], γ)

      @info "Saving checkpoint and seed cache"
      isfile(check_name) && run(`mv $check_name $check_name.backup`)
      save(check_name, Dict("gen"=>g, "gamma"=>γ, "pop"=>pop, 
                  "archive"=>archive, "BC"=> BC, "F"=>F,
                  "best"=>best, "novelties"=>novelties, 
                  "elites"=>elites, "sc"=>rm_params(sc)))

      global prefixes
      @info "computing prefixes"
      prefixes = compute_prefixes(elites)
      @info "distributing prefixes: $(prefixes)"
      @everywhere prefixes = $prefixes
    end
    if best[1] > 100
        @info "Returning: Best individal found with fitness $(best[1]) and BC $(best[3])"
        save("outs/$clsname/$expname/best.jld2", Dict("best"=>best))
        return
    end
    @info "Creating next pop"
    pop, elites = create_next_pop(g, sc, pop, F, novelties, BC, γ, args["num-elites"])
    cache_elites!(sc, mi, elites)
  end
end
