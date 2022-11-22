include("multiproc.jl")
using DataFrames
using CSV
using FileIO
using Infiltrator

@everywhere begin
  args = $args
  args["local"] && using Revise
  inc = args["local"] ? includet : include
  inc("es.jl")
  inc("net.jl")
  inc("trade.jl")
  inc("utils.jl")
end

expname = args["exp-name"]
@everywhere begin
  using .DistributedES
  using .Trade
  using .Net
  using .Utils
  using Flux
  using Statistics
  using StableRNGs


  env_config = mk_env_config(args)
  
  function run_batch(batch_size::Int, models::Dict{String,<:Chain}; evaluation=false, render_str::Union{Nothing,String}=nothing)

    b_env = [Trade.PyTrade.Trade(env_config) for _ in 1:batch_size]
    obs_size = (b_env[1].obs_size..., batch_size)
    num_actions = b_env[1].num_actions
    b_obs = batch_reset!(b_env, models)
    max_steps = args["episode-length"] * args["num-agents"]
    rews = Dict(key => 0.0f0 for key in keys(models))
    for _ in 1:max_steps
      b_obs, b_rew, b_dones, b_acts = batch_step!(b_env, models, b_obs, evaluation=evaluation)
      for (b, rew_dict) in enumerate(b_rew)
        for (name, rew) in rew_dict
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
    rew_dict, mets
  end

  function fitness_pos(p1::Int, p2::Int)
    models = Dict("f0a0" => re(θ .+ get_noise(nt, p1)),
      "f1a0" => re(θ .+ get_noise(nt, p2)))
    rew_dict, _ = run_batch(batch_size, models)
    rew_dict["f0a0"], rew_dict["f1a0"]
  end
  function fitness_neg(p1::Int, p2::Int)
    models = Dict("f0a0" => re(θ .- get_noise(nt, p1)),
      "f1a0" => re(θ .- get_noise(nt, p2)))
    rew_dict, _ = run_batch(batch_size, models)
    rew_dict["f0a0"], rew_dict["f1a0"]
  end

end

function main()

  dt_str = args["datime"]
  logname = "runs/$dt_str-$expname.log"
  println("$expname")
  df = nothing
  @everywhere begin
    opt = Adam(args["alpha"])
    pop_size = args["pop-size"]
    mut = args["mutation-rate"]
    α = args["alpha"]
    rng = StableRNG(0)
    env = Trade.PyTrade.Trade(env_config)
    batch_size = args["batch-size"]
    θ, re = make_model(Symbol(args["model"]), (env.obs_size..., batch_size), env.num_actions) |> Flux.destructure
    model_size = size(θ)[1]
  end
  
  # ###############
  # load checkpoint
  # ###############
  check_name = "outs/$expname/check.jld2"
  met_csv_name = "outs/$expname/metrics.csv"
  start_gen = 1
  # check if check exists on the file system
  if isfile(check_name)
    @assert isfile(met_csv_name)
    check = load(check_name)
    df = CSV.read(met_csv_name, DataFrame)
    @everywhere θ = $check["theta"]
    start_gen = check["gen"] + 1
    println("resuming from gen $start_gen")
  end

  for i in start_gen:args["num-gens"]

    if i % 3 == 0
      # compute and write metrics and outfile
      outdir = "outs/$expname/$i"
      run(`mkdir -p $outdir`)
      models = Dict("f0a0" => re(θ), "f1a0" => re(θ))
      rew_dict, mets = run_batch(batch_size, models, evaluation=false, render_str=outdir)
      df = update_df(df, mets)
      CSV.write(met_csv_name, df)

      # log mets and save gen
      avg_self_fit = round((rew_dict["f0a0"] + rew_dict["f1a0"]) / 2, digits=2)
      llog(islocal=args["local"], name=logname) do logfile
        println(logfile, "Generation $i: $avg_self_fit")
      end
      !args["local"] && save(check_name, Dict("theta"=>θ,"gen"=>i))

    end

    @everywhere begin
      nt = NoiseTable(rng, model_size, pop_size, mut)
    end

    # CHECK TO CONFIRM RNG IS SYNCHRONIZED
    rands = [fetch(remotecall(() -> nt.noise[1], p)) for p in 1:nprocs()]
    @assert length(unique(rands)) == 1


    fut_pos = []
    fut_neg = []

    for p1 in 1:pop_size
      #for p2 in 1:pop_size
      p2 = p1
      push!(fut_pos, remotecall(() -> fitness_pos(p1, p2), procs()[(p2%nprocs())+1]))
      push!(fut_neg, remotecall(() -> fitness_neg(p1, p2), procs()[(p2%nprocs())+1]))
    end

    fut_pos = [fetch(f)[1] for f in fut_pos]
    fut_neg = [fetch(f)[1] for f in fut_neg]

    # Log fitness distribution
    llog(islocal=args["local"], name=logname) do logfile
      println(logfile, "min=$(min(fut_pos...)) mean=$(mean(fut_pos)) max=$(max(fut_pos...)) std=$(std(fut_pos))")
    end

    F = fut_pos .- fut_neg
    @assert length(F) == pop_size
    ranks = Float32.(compute_centered_ranks(F))

    @everywhere begin
      grad = (args["l2"] * θ) - compute_grad(nt, $ranks) / (pop_size * mut)
      Flux.Optimise.update!(opt, θ, grad)
      @assert θ .|> isnan |> any |> !
    end
  end
end