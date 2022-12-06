using CSV
using Distributed
using DataFrames
using FileIO
using Infiltrator

@everywhere begin
    using EvoTrade
    args = $args
    env_type = !isnothing(args["maze"]) ? Val(:maze) : Val(:trade)
    expname = args["exp-name"]
    using Flux
    using Statistics
    using StableRNGs

    env_config = mk_env_config(args)


    function fitness(p1::T, p2::T) where T<:Vector{<:UInt32}
        models = Dict("f0a0" => re(reconstruct(p1, model_size, args["mutation-rate"])),
        "f1a0" => re(reconstruct(p2, model_size, args["mutation-rate"])))
        rew_dict, _, bc = run_batch(env_type, args["batch-size"], models)
        rew_dict["f0a0"], rew_dict["f1a0"], bc["f0a0"], bc["f1a0"]
    end


    function run_batch(::Val{:maze}, batch_size::Int, models::Dict{String,<:Chain}; evaluation=false, render_str::Union{Nothing,String}=nothing)

        reset!(env)
        r = -Inf
        for i in 1:args["episode-length"]
            obs = get_obs(env)
            probs = models["f0a0"](obs)
            acts = sample_batch(probs)
            @assert length(acts) == 1
            r, done = step!(env, acts[1])
            done && break
        end
        rews = Dict("f0a0" => r, "f1a0"=> r)
        bc = Dict("f0a0" => env.locations[4], "f1a0"=> env.locations[4])
        rews, nothing, bc
    end

    function run_batch(::Val{:trade}, batch_size::Int, models::Dict{String,<:Chain}; evaluation=false, render_str::Union{Nothing,String}=nothing)

        b_env = [PyTrade().Trade(env_config) for _ in 1:batch_size]
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
        env = !isnothing(args["maze"]) ? maze_from_file(args["maze"]) : PyTrade().Trade(env_config)
        θ, re = make_model(Symbol(args["model"]),
                (env.obs_size..., args["batch-size"]),
                env.num_actions,
                vbn=args["algo"]=="es",
                lstm=args["lstm"]) |> Flux.destructure
        model_size = length(θ)
    end

    pop = [[rand(UInt32)] for _ in 1:pop_size]
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
        df = isfile(met_csv_name) ? CSV.read(met_csv_name, DataFrame) : nothing
        check = load(check_name)
        start_gen = check["gen"] + 1
        F = check["F"]
        BC = check["BC"]
        best = check["best"]
        archive = check["archive"]
        pop = create_next_pop(start_gen, check["pop"], args["num-elites"])
        ts("resuming from gen $start_gen")
    end

    for g in start_gen:args["num-gens"]

        i₀ = g==1 ? 1 : 2

        ts("pmapping")
        fetches = pmap(i₀:pop_size) do p
            fitness(pop[p], pop[p])
        end


        if g==1
            F = [(fet[1]+fet[2])/2 for fet in fetches]
            BC = [fet[3] for fet in fetches]
        else
            F  = vcat([F[1]],  [fet[1]+fet[2]/2 for fet in fetches])
            BC = vcat([BC[1]], [fet[3] for fet in fetches])
        end

        ts("computing elite by re-evaluating top performers")
        @assert length(F) == length(BC) == pop_size
        top_F_idxs = sortperm(F, rev=true)[1:min(10, pop_size)]
        @assert F[top_F_idxs[1]] >= F[top_F_idxs[2]]
        num_evals = 30
        rollout_Fs = pmap(1:10*num_evals) do rollout_idx
            # get member ∈ [1,10] from rollout count
            p = floor(Int, (rollout_idx-1) / num_evals) + 1
            @assert p in 1:10
            fit = fitness(pop[top_F_idxs[p]], pop[top_F_idxs[p]])
            fit[1] + fit[2]/2
        end
        @assert rollout_Fs isa Vector{<:AbstractFloat}
        accurate_Fs = [sum(rollout_Fs[i:i+num_evals-1])/num_evals for i in 1:num_evals:length(rollout_Fs)]
        @assert length(accurate_Fs) == 10
        best_gen_idx = argmax(accurate_Fs)
        best_gen = maximum(accurate_Fs), pop[top_F_idxs[best_gen_idx]]

        if best_gen[1] > best[1]
            llog(islocal=args["local"], name=logname) do logfile
                ts(logfile, "New best ind found, F=$(best_gen[1])")
            end
            best = best_gen
        end
        @assert best[1] >= maximum(accurate_Fs)
        
        add_to_archive!(archive, BC, pop)

        pop_and_arch_bc = vcat([bc for (bc, _) in archive], BC)
        @assert length(pop_and_arch_bc) == length(archive) + pop_size
        novelties = [compute_novelty(bc, pop_and_arch_bc, k=min(pop_size-1, 25)) for bc in BC]
        @assert length(novelties) == pop_size

        reorder!(novelties, F, BC, pop)

        # LOG
        if g % 1 == 0
            ts("log start")
            models = Dict("f0a0" => re(reconstruct(best[2], model_size, args["mutation-rate"])),
            "f1a0" => re(reconstruct(best[2], model_size, args["mutation-rate"])))

            # Compute and write metrics
            outdir = "outs/$expname/$g"
            run(`mkdir -p $outdir`)
            rew_dict, mets, _ = run_batch(env_type, args["batch-size"], models, evaluation=false, render_str=outdir)
            df = update_df(df, mets)
            write_mets(met_csv_name, df)

            # Log to file
            avg_self_fit = round((rew_dict["f0a0"] + rew_dict["f1a0"]) / 2; digits=2)
            llog(islocal=args["local"], name=logname) do logfile
                ts(logfile, "Generation $g: $avg_self_fit")
            end

            # Save checkpoint
            !args["local"] && save(check_name, Dict("gen"=>g, "pop"=>pop, "archive"=>archive, "BC"=> BC, "F"=>F, "best"=>best))

            if best[1] > 0 
                # TODO: if GAs can fetch food and return at night, then they
                # can get positive reward, and we will need to make this domain
                # specific
                best_bc = BC[argmax(F)]
                llog(islocal=args["local"], name=logname) do logfile
                    ts("Returning: Best individal found with fitness $(best[1]) and BC $best_bc")
                end
                save("outs/$expname/best.jld2", Dict("best"=>best, "bc"=>best_bc))
                return
            end
            ts("log end")
        end
        pop = create_next_pop(g, pop, args["num-elites"])
    end
end
