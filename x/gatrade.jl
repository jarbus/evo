using CSV
using Distributed
using DataFrames
using FileIO
using Infiltrator

@everywhere begin
    using EvoTrade
    args = $args
    expname = args["exp-name"]
    clsname = args["cls-name"]
    using Flux
    using Statistics
    using StableRNGs

    env_config = mk_env_config(args)

    function fitness(p1::T, p2::T) where T<:Vector{<:Float64}

        if p1 == p2
            params = reconstruct(sc, mi, p1)
            models = Dict("f0a0" => re(params), "f1a0" => re(params))
        else
            models = Dict("f0a0" => re(reconstruct(sc, mi, p1)),
            "f1a0" => re(reconstruct(sc, mi, p2)))
        end
        rew_dict, _, bc, infos = run_batch(env, models, args, evaluation=false)
        rew_dict["f0a0"], rew_dict["f1a0"], bc["f0a0"], bc["f1a0"], infos
    end
end

function main()
    dt_str = args["datime"]
    logname="runs/$clsname/$dt_str-$expname.log"
    println("cls: $clsname\nexp: $expname")
    df = nothing
    @everywhere begin
        pop_size = args["pop-size"]
        if !isnothing(args["maze"])
            table = nothing 
            env = maze_from_file(args["maze"])
        else
            env = PyTrade().Trade(env_config)
            EvoTrade.Trade.reset!(env)
            table = env.table 
        end

        m = make_model(Symbol(args["model"]),
            (env.obs_size..., args["batch-size"]),
            env.num_actions,
            vbn=args["algo"]=="es",
            lstm=args["lstm"])
        θ, re = Flux.destructure(m)
        mi = ModelInfo(m)
        model_size = length(θ)
        println("model has $model_size params")
        # pass mazeenv struct or trade config dict
        env = env isa MazeEnv ? env : env_config
        # nt = NoiseTable(StableRNG(123), model_size, args["pop-size"], 1f0)
        global sc = SeedCache(maxsize=args["num-elites"]*2)
    end

    pop = [Vector{Float64}([rand(UInt32)]) for _ in 1:pop_size]
    best = (-Inf, [])
    archive = Set()
    BC = nothing
    F = nothing
    γ = args["exploration-rate"]

    # ###############
    # load checkpoint
    # ###############
    check_name = "outs/$clsname/$expname/check.jld2"
    met_csv_name = "outs/$clsname/$expname/metrics.csv"
    sc_name = "outs/$clsname/$expname/sc.jld2"
    start_gen = 1
    # check if check exists on the file system
    if isfile(check_name)
        df = isfile(met_csv_name) ? CSV.read(met_csv_name, DataFrame) : nothing
        check = load(check_name)
        start_gen = check["gen"] + 1
        γ = check["gamma"]
        F, BC, best, archive, novelties = getindex.((check,), ["F", "BC", "best","archive", "novelties"])
        global sc = load(sc_name)["sc"]
        pop, elites = create_next_pop(start_gen-1, sc, check["pop"], F, novelties, BC, γ, args["num-elites"])
        @everywhere cache_elites!(sc, mi, $elites)
        ts("resuming from gen $start_gen")
    end

    for g in start_gen:args["num-gens"]

        llog(islocal=args["local"], name=logname) do logfile
            ts(logfile, "pmapping")
        end

        fetches = pmap(1:pop_size) do p
            fitness(pop[p], pop[p])
        end

        F = [(fet[1]+fet[2])/2 for fet in fetches]
        BC = [fet[3] for fet in fetches]
        walks::Vector{Vector{NTuple{2, Float64}}} = [fet[5]["avg_walks"]["f0a0"] for fet in fetches]
        walks = vcat(walks, [fet[5]["avg_walks"]["f1a0"] for fet in fetches])


        llog(islocal=args["local"], name=logname) do logfile
            ts(logfile, "computing elite by re-evaluating top performers")
        end
        @assert length(F) == length(BC) == pop_size
        elite = compute_elite(fitness, pop, F, k=args["num-elites"], n=30)

        # update elite and modify exploration rate
        Δγ = 0.02
        if elite[1] > best[1]
            γ = clamp(γ - Δγ, 0, 0.9)
            llog(islocal=args["local"], name=logname) do logfile
                ts(logfile, "New best ind found, F=$(elite[1]), γ decreased to $γ")
            end
            best = elite
        else
            γ = clamp(γ + Δγ, 0, 0.9)
            llog(islocal=args["local"], name=logname) do logfile
                ts(logfile, "no better elite found, increasing γ to $γ")
            end
        end
        
        add_to_archive!(archive, BC, pop, args["archive-prob"])

        bc_matrix = hcat(BC...)
        pop_and_arch = hcat([bc for (bc, _) in archive]..., bc_matrix)
        @assert size(pop_and_arch, 2) == pop_size + length(archive)
        @assert size(bc_matrix, 2) == pop_size
        @assert size(bc_matrix, 1) == size(BC[1], 1)
        # @assert

        llog(islocal=args["local"], name=logname) do logfile
            ts(logfile, "computing novelties")
        end
        novelties = compute_novelties(bc_matrix, pop_and_arch, k=min(pop_size-1, 25))
        @assert length(novelties) == pop_size

        # LOG
        if g % 1 == 0
            ts("log start")
            models = Dict("f0a0" => re(reconstruct(sc, mi, best[2])),
            "f1a0" => re(reconstruct(sc, mi, best[2])))

            # Compute and write metrics
            outdir = "outs/$clsname/$expname/$g"
            run(`mkdir -p $outdir`)
            plot_walks("$outdir/pop.png", table, walks)
            rew_dict, mets, _, _ = run_batch(env, models, args, evaluation=false, render_str=outdir)

            muts = g > 1 ? [mr(pop[i]) for i in 1:pop_size] : [0.0]
            mets["gamma"] = γ
            log_mmm(mets, "mutation_rate", muts)
            log_mmm(mets, "fitness", F)
            log_mmm(mets, "novelty", novelties)
            df = update_df(df, mets)
            write_mets(met_csv_name, df)

            # Log to file
            avg_self_fit = round((rew_dict["f0a0"] + rew_dict["f1a0"]) / 2; digits=2)
            llog(islocal=args["local"], name=logname) do logfile
                ts(logfile, "Generation $g: $avg_self_fit")
            end
            plot_bcs(outdir, env, BC, novelties)

            # Save checkpoint
            save(check_name, Dict("gen"=>g, "gamma"=>γ, "pop"=>pop, "archive"=>archive, "BC"=> BC, "F"=>F, "best"=>best, "novelties"=>novelties))
            ts("log end")
        end
        if best[1] > 5
            best_bc = BC[argmax(F)]
            llog(islocal=args["local"], name=logname) do logfile
                ts("Returning: Best individal found with fitness $(best[1]) and BC $best_bc")
            end
            save("outs/$clsname/$expname/best.jld2", Dict("best"=>best, "bc"=>best_bc))
            return
        end
        llog(islocal=args["local"], name=logname) do logfile
            ts(logfile, "cache_elites")
        end

        pop, elites = create_next_pop(g, sc, pop, F, novelties, BC, γ, args["num-elites"])
        @everywhere cache_elites!(sc, mi, $elites)
        # Save seed cache without parameters
        sc_no_params = SeedCache(maxsize=2*args["num-elites"])
        for (k,v) in sc
            sc_no_params[k] = Dict(ke=>ve for (ke,ve) in v if ke != :params)
        end
        save(sc_name, Dict("sc"=>sc_no_params))
    end

end
