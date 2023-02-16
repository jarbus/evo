using CSV
using Distributed
using DataFrames
using FileIO
using Infiltrator
using Logging

@everywhere begin
  using EvoTrade
  using Flux
  using Statistics
  using StableRNGs
  args = $args
  expname = args["exp-name"]
  clsname = args["cls-name"]
  env_config = mk_env_config(args)

  function fitness(group::Vector{RolloutInd}, eval_gen)
    dc = decompress_group(group, prefixes)
    models, id_map, rdc_mets = mk_mods(sc, mi, nt, dc,
                                no_cache=args["no-caching"])
    gamebatch = run_batch(env, models, args, evaluation=true)
    id_batch = process_batch(gamebatch, id_map, eval_gen)
    id_batch = EvoTrade.add_metrics(id_batch, rdc_mets)
    id_batch
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
  n_pops = isnothing(args["maze"]) ? 2 : 1
  pops = [Pop(string(i), args["pop-size"]) for i in 1:n_pops]
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
        group_fn = all_v_best
    end
    m = make_model(env, args)
    θ, re = Flux.destructure(m)
    mi = ModelInfo(m, re)
    model_size = length(θ)
    nt = NoiseTable(StableRNG(1), length(mi), args["mutation-rate"])
    # pass mazeenv struct or trade config dict
    env = env isa MazeEnv ? env : env_config
    global sc = SeedCache(maxsize=args["num-elites"]*3)
    prefixes = Dict{String, V32}()
  end
  @info "model has $model_size params"

  if isfile(check_name)
    @info "Loading from checkpoints"
    df = isfile(met_csv_name) ? CSV.read(met_csv_name, DataFrame) : nothing 
    check = try 
        load(check_name)
    catch 
        load(check_name*".backup") 
    end
    start_gen = check["gen"] + 1
    pops,γ,prefixes = getindex.((check,),["pops","gamma","prefixes"])
    global sc = check["sc"]
    @info "resuming from gen $start_gen"
  end

  for g in start_gen:args["num-gens"]
    global prefixes
    @info "starting generation $g"
    eval_gen = g % 20 == 1
    @info "compressing pop"
    # save start time to variable start
    gen_start = time()
    rollout_pops = compress_pops(pops, prefixes)

    bytes = compute_compression_data(rollout_pops, prefixes)
    @info "bytes.compressed: |$(bytes.compressed)|"
    @info "bytes.uncompressed: |$(bytes.uncompressed)|"
    @info "creating rollout groups"
    groups = group_fn(rollout_pops...,
                      rollout_group_size=args["rollout-group-size"],
                      rollouts_per_ind=args["rollout-groups-per-mut"])
    @info "pmapping"
    id_batches = pmap(wp, groups) do g
        fitness(g, eval_gen)
    end
    @info "updating population"
    update_pops!(pops, id_batches, args["archive-prob"])
    @info "Creating next pop"
    next_pops = create_next_pop(pops, γ, args["num-elites"])
    gen_end = time()
    @info "Genome_Lengths: $(mmms([ceil(Int, length(ind.geno)/2) for pop in pops for ind in pop.inds]))"
    @info "Time_Per_Generation: |$(round(gen_end - gen_start, digits=2))|"

    if eval_gen # collect data only on evaluation generations
      @info "log start"
      metrics_csv = Dict()
      metrics_csv["Time_Per_Generation"] = round(gen_end - gen_start, digits=2)
      outdir="outs/$clsname/$expname/"*string(g,pad=3,base=10)
      run(`mkdir -p $outdir`)

      @info "Running elite eval"
      rollout_elites = compress_elites(next_pops, prefixes)
      eval_groups = group_fn(rollout_elites..., 
                      rollout_group_size=args["rollout-group-size"],
                      rollouts_per_ind=args["rollout-groups-per-mut"])
      eval_metrics = pmap(wp, eval_groups) do group
        dc = decompress_group(group, prefixes)
        models, id_map, _ = mk_mods(sc, mi, nt, dc,
                                no_cache=args["no-caching"])
        model_names = models |> keys |> collect
        str_name = joinpath(outdir, string(hash(model_names))*"-"*string(myid()))
        gamebatch = run_batch(env, models, args, evaluation=true, render_str=str_name)
        id_batch = process_batch(gamebatch, id_map, eval_gen)
        id_batch.mets
      end |> aggregate_metrics
      @info "Logging metrics"
      rollout_metrics = aggregate_metrics(id_batches)
      for (met_name, met_vec) in rollout_metrics
          log_mmm!(metrics_csv, "pop_"*met_name, met_vec)
      end
      for (met_name, met_vec) in eval_metrics
          log_mmm!(metrics_csv, "eval_"*met_name, met_vec)
      end
      for pop in pops
        log_mmm!(metrics_csv, "fitness-$(pop.id)", fitnesses(pop))
        log_mmm!(metrics_csv, "novelty-$(pop.id)", novelties(pop))
        metrics_csv["archive-size-$(pop.id)"] = length(pop.archive)
      end
      metrics_csv["gamma"] = γ
      df = update_df_and_write_metrics(met_csv_name, df, metrics_csv)

      @info "Visualizing outs"
      #isnothing(args["maze"]) && vis_outs(outdir, args["local"])
      plot_grid_and_walks(env, "$outdir/pop", grid, pops, args["num-elites"], γ)
      # plot_bcs("$outdir/bcs", pops, args["num-elites"])


      if !args["no-compression"]
        global prefixes
        @info "computing prefixes"
        prefixes = compute_prefixes(pops)
        @everywhere prefixes = $prefixes
      end

      @info "Saving checkpoint and seed cache"
      isfile(check_name) && run(`mv $check_name $check_name.backup`)
      save(check_name, Dict("gen"=>g, "gamma"=>γ, "pops"=>next_pops,
                         "sc"=>rm_params(sc),"prefixes"=>prefixes))
    end
    pops = next_pops
  end
end
