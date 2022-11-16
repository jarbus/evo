include("multiproc.jl")
using Dates
using DataFrames
using CSV
using FileIO
using Infiltrator

@everywhere begin
  # Load args
  args = $args
  if !args["local"]
    include("ga.jl")
    include("net.jl")
    include("trade.jl")
  else
    using Revise
    includet("ga.jl")
    includet("net.jl")
    includet("trade.jl")
  end
end

expname = args["exp-name"]
@everywhere begin
  using .Trade
  using .Net
  using .GANS
  using Flux
  using Statistics
  using StableRNGs


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


  function fitness(p1::T, p2::T) where T<:Vector{<:UInt32}
    models = Dict("f0a0" => re(reconstruct(p1, model_size)),
                  "f1a0" => re(reconstruct(p2, model_size)))
    rew_dict, _, bc = run_batch(batch_size, models)
    rew_dict["f0a0"], rew_dict["f1a0"], bc["f0a0"], bc["f1a0"]
  end

  
  function run_batch(batch_size::Int, models::Dict{String,<:Chain}; evaluation=false, render_str::Union{Nothing,String}=nothing)

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
    mets = get_metrics(b_env[1])
    rew_dict, mets, total_acts
  end
end

function main()
  dt_str = Dates.format(now(), "mm-dd_HH:MM")
  println("$expname")
  df = nothing
  @everywhere begin
    pop_size = args["pop-size"]
    T = args["num-elites"]
    env = Trade.PyTrade.Trade(env_config)
    batch_size = args["batch-size"]
    θ, re = make_model(Symbol(args["model"]), (env.obs_size..., batch_size), env.num_actions) |> Flux.destructure
    #opt = Adam(args["alpha"])
    #mut = args["mutation-rate"]
    #α = args["alpha"]
    model_size = length(θ)
  end

  pop = Vector{Vector{UInt32}}()
  next_pop = Vector{Vector{UInt32}}()
  best = (-Inf, [])
  archive = Set{Tuple{Vector{Float32},Vector{UInt32}}}()
  BC = nothing
  F = nothing
  for g in 1:args["num-gens"]
    println("Running generation")

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
    println("call")
    for p1 in i₀:pop_size
      #for p2 in 1:pop_size
      p2 = p1
      push!(futs, remotecall(() -> fitness(pop[p1], pop[p2]), procs()[(p2%nprocs())+1]))
    end
    println("fetching")
    fetches = [fetch(fut) for fut in futs]
    println("fetched")
    if g==1
        F = [fet[1]+fet[2]/2 for fet in fetches]
        BC = [(fet[3] .+ fet[4])/2 for fet in fetches]
    else
        BC = vcat([BC[1]], [(fet[3].+fet[4])./2 for fet in fetches])
        F  = vcat([F[1]],  [fet[1]+fet[2]/2 for fet in fetches])
    end
    @assert length(F) == length(BC) == pop_size
    println(F)
    max_fit = max(F...)
    if max_fit > best[1]
        println("New best ind found, F=$max_fit")
        best = (max_fit, pop[argmax(F)])
    end
    for i in 1:pop_size
        if i > 1 && rand() > 0.01
            push!(archive, (BC[i], pop[i]))
        end
    end
    novelties = [compute_novelty(bc, archive) for bc in BC]
    order = sortperm(novelties, rev=true)
    pop = pop[order]
    BC = BC[order]
    F =  F[order]

    # LOG
    if g % 1 == 0
      outdir = "outs/$expname/$g"
      run(`mkdir -p $outdir`)
      logfile = !args["local"] ? open("runs/$dt_str-$expname.log", "a") : stdout
      print(logfile, "Generation $g: ")
      models = Dict("f0a0" => re(reconstruct(pop[1], model_size)),
          "f1a0" => re(reconstruct(pop[1], model_size)))
      rew_dict, mets, _ = run_batch(batch_size, models)
      if isnothing(df)
        df = DataFrame(mets)
      else
        push!(df, mets)
      end
      !args["local"] && save("outs/$expname/models.jld2", models)
      CSV.write("outs/$expname/metrics.csv", df)
      avg_self_fit = (rew_dict["f0a0"] + rew_dict["f1a0"]) / 2
      println(logfile, "$(round(avg_self_fit, digits=2)) ")
      !args["local"] && close(logfile)
    end
  end

end
