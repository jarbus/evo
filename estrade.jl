include("multiproc.jl")
using Dates
using DataFrames
using CSV
using FileIO
using Infiltrator

@everywhere begin
    ts() = Dates.format(now(), "HH:MM:SS")*" "
    args = $args
    args["local"] && using Revise
    inc = args["local"] ? includet : include
    inc("es.jl")
    inc("net.jl")
    inc("trade.jl")
    inc("utils.jl")
    inc("maze.jl")
    @enum Env trade maze
    env_type = !isnothing(args["maze"]) ? Val(maze) : Val(trade)
end

expname = args["exp-name"]
@everywhere begin
    using .DistributedES
    using .Trade
    using .Net
    using .Utils
    using .Maze
    using Flux
    using Statistics
    using StableRNGs



    env_config = mk_env_config(args)

    function run_batch(::Val{trade}, batch_size::Int, models::Dict{String,<:Chain}; evaluation=false, render_str::Union{Nothing,String}=nothing)

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
            rew_dict, mets, nothing
    end

    function run_batch(::Val{maze}, batch_size::Int, models::Dict{String,<:Chain}; evaluation=false, render_str::Union{Nothing,String}=nothing)

        reset!(env)
        r = -Inf
        for i in 1:args["episode-length"]
            obs = get_obs(env)
            obs = repeat(obs, outer=(1,1,1,batch_size))
            probs = models["f0a0"](obs)
            acts = sample_batch(probs)
            @assert length(acts) == batch_size
            r, done = step!(env, acts[1])
            done && break
        end
        rews = Dict("f0a0" => r, "f1a0"=> r)
        bc = Dict("f0a0" => env.location, "f1a0"=> env.location)
        rews, nothing, bc
    end



    function fitness_pos(p1::Int, p2::Int)

        models = Dict("f0a0" => re(θ .+ get_noise(nt, p1)),
        "f1a0" => re(θ .+ get_noise(nt, p2)))
        rew_dict, _, _ = run_batch(env_type, batch_size, models)
        rew_dict["f0a0"], rew_dict["f1a0"]
    end
    function fitness_neg(p1::Int, p2::Int)
        models = Dict("f0a0" => re(θ .- get_noise(nt, p1)),
        "f1a0" => re(θ .- get_noise(nt, p2)))
        rew_dict, _, _ = run_batch(env_type, batch_size, models)
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
        if !isnothing(args["maze"])
            env = maze_from_file(args["maze"])
        else
            env = Trade.PyTrade.Trade(env_config)
        end

        batch_size = args["batch-size"]
        θ, re = make_model(args["model"]|>Symbol|>Val, (env.obs_size..., batch_size), env.num_actions) |> Flux.destructure
        model_size = length(θ)
    end

    # ###############
    # load checkpoint
    # ###############
    check_name = "outs/$expname/check.jld2"
    met_csv_name = "outs/$expname/metrics.csv"
    start_gen = 1
    # check if check exists on the file system
    if isfile(check_name)
        check = load(check_name)
        if isfile(met_csv_name)
            df = CSV.read(met_csv_name, DataFrame)
        end
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
            rew_dict, mets, _ = run_batch(env_type, batch_size, models, evaluation=false, render_str=outdir)
            if !isnothing(mets)
                df = update_df(df, mets)
                CSV.write(met_csv_name, df)
            end

            # log mets and save gen
            avg_self_fit = round((rew_dict["f0a0"] + rew_dict["f1a0"]) / 2, digits=2)
            llog(islocal=args["local"], name=logname) do logfile
                println(logfile, ts() * "Generation $i: $avg_self_fit")
            end
            !args["local"] && save(check_name, Dict("theta"=>θ,"gen"=>i))
            # TODO: remove if we can fix this
            if avg_self_fit > 0
                println("Returning: eval fitness > 0")
                return
            end

        end

        @everywhere begin
            nt = NoiseTable(rng, model_size, pop_size, mut)
        end

        # CHECK TO CONFIRM RNG IS SYNCHRONIZED
        rands = [fetch(remotecall(() -> nt.noise[1], p)) for p in 1:nprocs()]
        @assert length(unique(rands)) == 1

        # run fitness_pos and fitness_neg in parallel
        println("pmapping")
        futs = pmap(1:pop_size) do p1
            println(p1)
            p2 = p1
            fitness_pos(p1, p2)[1], fitness_neg(p1, p2)[1]
        end
        println("pmapped")
        fut_pos = [f[1] for f in futs]
        fut_neg = [f[2] for f in futs]


        # Log fitness distribution
        if i % 3 == 0
            llog(islocal=args["local"], name=logname) do logfile
                println(logfile, ts()*"min=$(min(fut_pos...)) mean=$(mean(fut_pos)) max=$(max(fut_pos...)) std=$(std(fut_pos))")
                println(logfile, ts()*"min=$(min(fut_neg...)) mean=$(mean(fut_neg)) max=$(max(fut_neg...)) std=$(std(fut_neg))")
            end
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
