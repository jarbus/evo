include("multiproc.jl")
using DataFrames
using CSV
using FileIO
using Infiltrator

@everywhere begin
  args = $args
  args["local"] && using Revise
  inc = args["local"] ? includet : include
  inc("ga.jl")
  inc("net.jl")
  inc("trade.jl")
  inc("utils.jl")
  inc("maze.jl")

  @enum Env trade maze
  env_type = !isempty(args["maze"]) ? maze : trade
  
end

expname = args["exp-name"]
@everywhere begin
  using .Trade
  using .Net
  using .GANS
  using .Utils
  using .Maze
  using Flux
  using Statistics
  using StableRNGs


  env_config = mk_env_config(args)


  function fitness(p1::T, p2::T) where T<:Vector{<:UInt32}
    models = Dict("f0a0" => re(reconstruct(p1, model_size, 0.01)),
                  "f1a0" => re(reconstruct(p2, model_size, 0.01)))
    rew_dict, _, bc = run_batch(Val(env_type), batch_size, models)
    rew_dict["f0a0"], rew_dict["f1a0"], bc["f0a0"], bc["f1a0"]
  end

  
  function run_batch(::Val{maze}, batch_size::Int, models::Dict{String,<:Chain}; evaluation=false, render_str::Union{Nothing,String}=nothing)

    reset!(env)
    total_rew = 0
    for i in 1:args["episode-length"]
        obs = get_obs(env)
        probs = models["f0a0"](obs)
        acts = sample_batch(probs)
        @assert length(acts) == 1
        r, done = step!(env, acts[1])
        total_rew += r
    end
    rews = Dict("f0a0" => total_rew, "f1a0"=> total_rew)
    bc = Dict("f0a0" => env.location, "f1a0"=> env.location)
    rews, nothing, bc
  end

  function run_batch(::Val{trade}, batch_size::Int, models::Dict{String,<:Chain}; evaluation=false, render_str::Union{Nothing,String}=nothing)

    b_env = [Trade.PyTrade.Trade(env_config) for _ in 1:batch_size]
    obs_size = (b_env[1].obs_size..., batch_size)
    num_actions = b_env[1].num_actions
    b_obs = batch_reset!(b_env, models)
    max_steps = args["episode-length"] * args["num-agents"]
    rews = Dict(key => 0.0f0 for key in keys(models))
    total_acts = Dict(key => Vector{UInt32}() for key in keys(models))
    for _ in 1:max_steps
      b_obs, b_rew, b_dones, b_acts = batch_step!(b_env, models, b_obs, evaluation=evaluation)
      for (b, rew_dict) in enumerate(b_rew)
        for (name, rew) in rew_dict

          total_acts[name] = vcat(total_acts[name], b_acts)
          rews[name] += rew

          if render_str isa String && name == first(models).first
            renderfile = "$render_str/b$b.out"
            render(b_env[b], renderfile)
          end
        end
      end
    end
    rew_dict = Dict(name => rew / batch_size for (name, rew) in rews)
    mets = get_metrics(b_env)
    bc = Dict(name => bc1(total_acts[name], num_actions) for (name, _) in models)
    rew_dict, mets, bc
  end
end

function main()
  dt_str = args["datime"]
  logname="runs/$dt_str-$expname.log"
  println("$expname")
  df = nothing
  @everywhere begin
    pop_size = args["pop-size"]
    T = args["num-elites"]
    mut = args["mutation-rate"]
    if "maze" in keys(args)
        env = maze_from_file(args["maze"])
    else
        env = Trade.PyTrade.Trade(env_config)
    end
    batch_size = args["batch-size"]
    θ, re = make_model(Symbol(args["model"]), (env.obs_size..., batch_size), env.num_actions, false) |> Flux.destructure
    model_size = length(θ)
  end

  pop = Vector{Vector{UInt32}}()
  next_pop = Vector{Vector{UInt32}}()
  best = (-Inf, [])
  archive = Set()
  BC = nothing
  F = nothing

  # ###############
  # load checkpoint
  # ###############
  check_name = "outs/$expname/check.jld2"
  met_csv_name = "outs/$expname/metrics.csv"
  start_gen = 1
  # check if check exists on the file system
  if isfile(check_name)
    @assert isfile(met_csv_name)
    df = CSV.read(met_csv_name, DataFrame)
    check = load(check_name)
    start_gen = check["gen"] + 1
    F = check["F"]
    BC = check["BC"]
    best = check["best"]
    archive = check["archive"]
    pop = check["pop"]
    next_pop = copy(pop)
    @assert length(pop) == pop_size
    println("resuming from gen $start_gen")
  end

  for g in start_gen:args["num-gens"]

    i₀ = g==1 ? 1 : 2
    # run on first gen
    for i in i₀:pop_size
      if g == 1
        push!(next_pop, [rand(UInt32)])
      else
        k = (rand(UInt) % T) + 1 # select parent
        next_pop[i] = copy(pop[k])
        push!(next_pop[i], rand(UInt32))
      end
    end
    @assert length(next_pop) == pop_size
    pop = next_pop 

    futs = []
    for p1 in i₀:pop_size
      p2 = p1
      push!(futs, remotecall(() -> fitness(pop[p1], pop[p2]), procs()[(p2%nprocs())+1]))
    end
    println("fetching")
    fetches = [fetch(fut) for fut in futs]
    println("fetched")
   
    if g==1
        F = [(fet[1]+fet[2])/2 for fet in fetches]
        BC = [fet[3] for fet in fetches]
    else
        F  = vcat([F[1]],  [fet[1]+fet[2]/2 for fet in fetches])
        BC = vcat([BC[1]], [fet[3] for fet in fetches])
    end

    @assert length(F) == length(BC) == pop_size
    max_fit = max(F...)
    if max_fit > best[1]
        llog(islocal=args["local"], name=logname) do logfile
            println(logfile, "New best ind found, F=$max_fit")
        end
        best = (max_fit, pop[argmax(F)])
    end
    for i in 1:pop_size
        if i > 1 && rand() > 0.01
            push!(archive, (BC[i], pop[i]))
        end
    end

    pop_and_arch_bc = vcat([bc for (bc, _) in archive], BC)
    @assert length(pop_and_arch_bc) == length(archive) + pop_size
    novelties = [compute_novelty(bc, pop_and_arch_bc) for bc in BC]
    @assert length(novelties) == pop_size

    order = sortperm(novelties, rev=true)
    pop = pop[order]
    BC = BC[order]
    F =  F[order]
    @assert length(pop) == length(BC) == length(F) == pop_size

    # LOG
    if g % 1 == 0
      models = Dict("f0a0" => re(reconstruct(pop[1], model_size)),
          "f1a0" => re(reconstruct(pop[1], model_size)))

      # Compute and write metrics
      outdir = "outs/$expname/$g"
      run(`mkdir -p $outdir`)
      rew_dict, mets, _ = run_batch(Val(env_type), batch_size, models, evaluation=false, render_str=outdir)
      if !isnothing(mets)
        df = update_df(df, mets)
        CSV.write(met_csv_name, df)
      end

      # Log to file
      avg_self_fit = round((rew_dict["f0a0"] + rew_dict["f1a0"]) / 2; digits=2)
      llog(islocal=args["local"], name=logname) do logfile
        println(logfile, "Generation $g: $avg_self_fit")
      end

      # Save checkpoint
      !args["local"] && save(check_name, Dict("gen"=>g, "pop"=>pop, "archive"=>archive, "BC"=> BC, "F"=>F, "best"=>best))
    end
  end
end
