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

    mets_to_return = vcat(
        [["$(met)_$f" for f in 0:args["food-types"]-1 for met in
            ("exchange", "picks", "places", "rew_light")]]...,
        [["strat_$s" for s in ("noop","give","take","exchange")]]...,
        ["rew_base_health"], ["rew_acts"], ["mut_exchanges"],
        ["gives"], ["takes"])
    env_config = mk_env_config(args)

    function fitness(group::Vector, eval_gen)
        counts = Dict{Int, Int}()
        for (i, _) in group
            counts[i] = get(counts, i, 0) + 1
        end
        group = decompress_group(group, prefixes)
        # creates mapping of pid_copy to params
        models = Dict(aid(i, c)=>re(reconstruct(sc, mi, seeds, e_idxs)) for (i, seeds, e_idxs) in group for c in 1:counts[i])
        rew_dict, mets, bc_dict, info_dict = run_batch(env, models, args, evaluation=true)
        rews, bcs, infos = Dict(), Dict(), Dict{Any, Any}("avg_walks"=>Dict())
        for (i, _) in group 
            i < 0 && continue # skip elites
            a_id = aid(i, 1)
            rews[i] = [rew_dict[a_id]]
            bcs[i] = [bc_dict[a_id]]
            infos["avg_walks"][i] = [info_dict["avg_walks"][a_id]]
            for c in 2:counts[i]
                a_id = aid(i, c)
                push!(rews[i], rew_dict[a_id])
                push!(bcs[i], bc_dict[a_id])
                push!(infos["avg_walks"][i], info_dict["avg_walks"][a_id])
            end
            if !eval_gen
                infos["avg_walks"][i] = []
            end
        end
        for (i, _) in group
            i < 0 && continue
            @assert i in keys(rews)
        end
        infos["mets"] = eval_gen ? filter(p->p.first in mets_to_return, mets) : []
        rews, bcs, infos
    end
end

function main()
  dt_str = args["datime"]
  logname="runs/$clsname/$dt_str-$expname.log"
  check_name = "outs/$clsname/$expname/check.jld2"
  met_csv_name = "outs/$clsname/$expname/metrics.csv"
  sc_name = "outs/$clsname/$expname/sc.jld2"
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
  global metrics

  @info "cls: $clsname"
  @info "exp: $expname"
  @info "Running on commit: "*read(`git rev-parse --short HEAD`, String)
  
  @info "Initializing on all workers"
  @everywhere begin
    if !isnothing(args["maze"])
        env = maze_from_file(args["maze"])
        grid = env.grid 
    else
        env = PyTrade().Trade(env_config)
        EvoTrade.Trade.reset!(env)
        grid = env.table 
    end

    m = make_model(Symbol(args["model"]),
        (env.obs_size..., args["batch-size"]),
        env.num_actions,
        vbn=args["algo"]=="es",
        lstm=args["lstm"])
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
    # We can get pre-empted while checkpointing, meaning the 
    # latest check might be invalid. We always mv the 
    # previous jld2 files to *.jld2-old, and boot from that
    # if the latest check is corrupt.
    try
        global check = load(check_name)
        global sc = load(sc_name)["sc"]
    catch
        global check = load(check_name*"-old")
        global sc = load(sc_name*"-old")["sc"]
    end
    start_gen = check["gen"] + 1
    γ = check["gamma"]
    F, BC, best, archive, novelties = getindex.((check,), ["F", "BC", "best","archive", "novelties"])
    global pop = check["pop"]
    global elites = check["elites"]

    cache_elites!(sc, mi, elites)

    @info "resuming from gen $start_gen"
  end

  for g in start_gen:args["num-gens"]
    eval_gen = true #g % 25 == 1
    global prefixes
    @info "starting generation $g"
    @info "creating groups"

    @info "compressing pop"
    rollout_pop = compress_pop(pop, elites, prefixes)
    @info "creating rollout groups"
    groups = all_v_all(rollout_pop)

    @info "pmapping"
    fetches = pmap(wp, groups) do g
        fitness(g, eval_gen)
    end
    
    F, BC, walks, rollout_metrics = aggregate_rollouts(fetches, pop_size)

    @assert length(F) == length(BC) == pop_size
    elite = (maximum(F), pop[argmax(F)])

    # update elite and modify exploration rate
    if elite[1] > best[1]
        γ = 0.1
        @info "New best ind found, F=$(elite[1]), γ decreased to $γ"
        best = elite
    else
        # TODO change gamma to clamped value once GA test passes
        #γ = clamp(γ + 0.02, 0, 0.9)
        γ = 0.1
        @info "no better elite found, set γ to $γ"
    end
    
    add_to_archive!(archive, BC, pop, args["archive-prob"])

    bc_matrix = hcat(BC...)
    pop_and_arch = hcat([bc for (bc, _) in archive]..., bc_matrix)
    @assert size(pop_and_arch, 2) == pop_size + length(archive)
    @assert size(bc_matrix, 2) == pop_size
    @assert size(bc_matrix, 1) == size(BC[1], 1)

    @info "computing novelties"
    novelties = compute_novelties(bc_matrix, pop_and_arch, k=min(pop_size-1, 25))
    @assert length(novelties) == pop_size
    @info "most novel bc: $(BC[argmax(novelties)])"
    @info "most fit bc: $(BC[argmax(F)]), fitness $(maximum(F))"

    # Run evaluation, collect statistics, and checkpoint every 50 gens
    if eval_gen
      @info "keys of final rollout mets: $(keys(rollout_metrics))"

      prefixes = compute_prefixes(elites)
      @info "distributing prefixes: $(prefixes)"
      @everywhere prefixes = $prefixes

      # @spawnat 1 begin
      begin
        @info "log start"
        metrics_csv = Dict()

        outdir = "outs/$clsname/$expname/"*string(g, pad=3, base=10)
        run(`mkdir -p $outdir`)

        plot_grid_and_walks(env, "$outdir/pop.png", grid, walks, novelties, F, args["num-elites"], γ)

        eval_members = rollout_pop[sortperm(F, rev=true)[1:ceil(Int, args["num-elites"]*(1-γ))]]
        eval_groups = all_v_all(eval_members)
        eval_metrics = pmap(wp, eval_groups) do group
           group = decompress_group(group, prefixes)
           models = Dict("p$i" => re(reconstruct(sc, mi, seeds, eidxs)) for (i, seeds, eidxs) in group)
           str_name = joinpath(outdir, string(hash(group))*"-"*string(myid()))
           _, metrics, _, _ = run_batch(env, models, args, evaluation=true, render_str=str_name, batch_size=2)
           metrics
        end |> aggregate_metrics
        @info "Logging evaluation metrics"
        for (met_name, met_vec) in eval_metrics
            @info "Logging $met_name with length $(length(met_vec))"
            log_mmm!(metrics_csv, "eval_"*met_name, met_vec)
        end
        @info "Visualizing outs"
        isnothing(args["maze"]) && vis_outs(outdir, args["local"])

        @info "Logging population metrics"
        global rollout_metrics
        for (met_name, met_vec) in rollout_metrics
            log_mmm!(metrics_csv, "pop_"*met_name, met_vec)
        end
        log_mmm!(metrics_csv, "fitness", F)
        log_mmm!(metrics_csv, "novelty", novelties)
        metrics_csv["gamma"] = γ
        @info "writing mets"
        update_df_and_write_metrics(met_csv_name, df, metrics_csv)

        # Save checkpoint
        @info "savefile"
        isfile(check_name) && run(`mv $check_name $check_name-old`)
        save(check_name, Dict("gen"=>g, "gamma"=>γ, "pop"=>pop, 
                        "archive"=>archive, "BC"=> BC, "F"=>F,
                        "best"=>best, "novelties"=>novelties, "elites"=>elites))

        # Save seed cache without parameters
        isfile(sc_name) && run(`mv $sc_name $sc_name-old`)
        save_sc(sc_name, sc)
      end
    end

    if best[1] > 100
        best_bc = BC[argmax(F)]
        @info "Returning: Best individal found with fitness $(best[1]) and BC $best_bc"
        save("outs/$clsname/$expname/best.jld2", Dict("best"=>best, "bc"=>best_bc))
        return
    end

    pop, elites = create_next_pop(g, sc, pop, F, novelties, BC, γ, args["num-elites"])
    @info "cache_elites"
    cache_elites!(sc, mi, elites)
  end
end
