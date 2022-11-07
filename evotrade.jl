using Distributed
using Dates
using DataFrames
using CSV

if !args["local"]
  function get_procs(str)
    full_cpus_per_node = Vector{Int}()
    for c in str
      m = match(r"(\d+)\(x(\d+)\)", c)
      if !isnothing(m)
        for i in 1:parse(Int, m[2])
          push!(full_cpus_per_node, parse(Int, m[1]))
        end
      else
        push!(full_cpus_per_node, parse(Int, c))
      end
    end
    full_cpus_per_node
  end

  cpus_per_node = get_procs(split(ENV["SLURM_JOB_CPUS_PER_NODE"], ","))
  nodelist = ENV["SLURM_JOB_NODELIST"]
  hostnames = read(`scontrol show hostnames "$nodelist"`, String) |> strip |> split .|> String
  @assert length(cpus_per_node) == length(hostnames)

  machine_specs = [hostspec for hostspec in zip(hostnames, cpus_per_node)]
  println(machine_specs)
  addprocs(machine_specs, max_parallel=100, multiplex=true)
  println("nprocs $(nprocs())")
else
  addprocs(2)
end

@everywhere begin
  include("es.jl")
  include("trade.jl")
end

expname = args["exp-name"]
@everywhere begin
  args = $args
  using .DistributedES
  using .Trade
  using Flux
  using Statistics
  using StableRNGs

  # Load args

  env_config = Dict(
    "window" => args["window"],
    "grid" => (args["gx"], args["gy"]),
    "food_types" => 2,
    "latest_agent_ids" => [0, 0],
    "matchups" => [("f0a0", "f1a0")],
    "episode_length" => args["episode-length"],
    "respawn" => true,
    "fires" => [Tuple(args["fires"][i:i+1]) for i in 1:2:length(args["fires"])],
    "foods" => [Tuple(args["foods"][i:i+2]) for i in 1:3:length(args["foods"])],
    "health_baseline" => true,
    "spawn_agents" => "center",
    "spawn_food" => "corner",
    "light_coeff" => 10,
    "food_agent_start" => args["food-agent-start"],
    "food_env_spawn" => args["food-env-spawn"],
    "day_night_cycle" => true,
    "day_steps" => args["day-steps"],
    "vocab_size" => 0)

  function run_batch(batch_size::Int, models::Dict{String,<:Chain}; evaluation=false, render_str::Union{Nothing,String}=nothing)

    b_env = [Trade.PyTrade.Trade(env_config) for _ in 1:batch_size]
    obs_size = (b_env[1].obs_size..., batch_size)
    num_actions = b_env[1].num_actions
    b_obs = batch_reset!(b_env, models)
    max_steps = args["episode-length"] * args["num-agents"]
    rews = Dict(key => 0.0f0 for key in keys(models))
    for _ in 1:max_steps
      b_obs, b_rew, b_dones = batch_step!(b_env, models, b_obs, evaluation=evaluation)
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
    mets = get_metrics(b_env[1])
    rew_dict, mets
  end

end

function compute_matrix_fitness(A::Matrix{Tuple{Float32,Float32}}, i::Integer)
  row_fit = sum([e[1] for e in A[i, :]])
  col_fit = sum([e[2] for e in A[:, i]])
  row_fit + col_fit / (size(A, 1)^2)
end

function main()

  dt_str = Dates.format(now(), "mm-dd_HH:MM")
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
    base_model = make_model(:small, (env.obs_size..., batch_size), env.num_actions)
    θ, re = Flux.destructure(base_model)
    model_size = size(θ)[1]
  end

  for i in 1:args["num-gens"]

    if i % 1 == 0
      outdir = "outs/$expname/$i"
      run(`mkdir -p $outdir`)
      logfile = !args["local"] ? open("runs/$dt_str-$expname.log", "a") : stdout
      print(logfile, "Generation $i: ")
      rew_dict, mets = run_batch(4, Dict("f0a0" => re(θ),
          "f1a0" => re(θ)), evaluation=true, render_str=outdir)
      if isnothing(df)
        df = DataFrame(mets)
      else
        push!(df, mets)
      end
      CSV.write("outs/$expname/metrics.csv", df)
      avg_self_fit = (rew_dict["f0a0"] + rew_dict["f1a0"]) / 2
      println(logfile, "$(round(avg_self_fit, digits=2)) ")
      !args["local"] && close(logfile)
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
      push!(fut_pos, remotecall(procs()[(p2%nprocs())+1]) do
        models = Dict("f0a0" => re(θ .+ get_noise(nt, p1)),
          "f1a0" => re(θ .+ get_noise(nt, p2)))
        rew_dict, _ = run_batch(batch_size, models)
        rew_dict["f0a0"], rew_dict["f1a0"]
      end)
      push!(fut_neg, remotecall(procs()[(p2%nprocs())+1]) do
        models = Dict("f0a0" => re(θ .- get_noise(nt, p1)),
          "f1a0" => re(θ .- get_noise(nt, p2)))
        rew_dict, _ = run_batch(batch_size, models)
        rew_dict["f0a0"], rew_dict["f1a0"]
      end)
    end

    F = [fetch(f)[1] for f in fut_pos] .- [fetch(f)[1] for f in fut_neg]

    logfile = !args["local"] ? open("runs/$dt_str-$expname.log", "a") : stdout
    println(logfile, "min=$(min(F...)) mean=$(mean(F)) max=$(max(F...)) std=$(std(F))")
    !args["local"] && close(logfile)
    #fits = reshape(fits, (pop_size, pop_size))
    #fits = [compute_matrix_fitness(fits, i) for i in 1:pop_size]
    @assert length(F) == pop_size
    ranks = Float32.(compute_centered_ranks(F))

    @everywhere begin
      grad = compute_grad(nt, $ranks) / (pop_size * mut)
      Flux.Optimise.update!(opt, θ, -grad)
    end
    #ranks /= length(ranks)
    # A = (fits .- mean(fits)) ./ (std(fits) + 0.0001f0)
    #if A == zeros(size(A))
    #  A = ones(length(A)) / length(A)
    #end
  end
end
