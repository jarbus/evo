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

    function fitness(p1::T, p2::T) where T<:Vector{<:UInt32}

        if p1 == p2
            params = reconstruct(sc, mi, p1, args["mutation-rate"])
            models = Dict("f0a0" => re(params), "f1a0" => re(params))
        else
            models = Dict("f0a0" => re(reconstruct(sc, mi, p1, args["mutation-rate"])),
            "f1a0" => re(reconstruct(sc, mi, p2, args["mutation-rate"])))
        end
        rew_dict, _, bc = run_batch(env, models, args, evaluation=false)
        rew_dict["f0a0"], rew_dict["f1a0"], bc["f0a0"], bc["f1a0"]
    end
end

function main()
    dt_str = args["datime"]
    logname="runs/$clsname/$dt_str-$expname.log"
    println("cls: $clsname\nexp: $expname")
    df = nothing
    @everywhere begin
        pop_size = args["pop-size"]
        env = !isnothing(args["maze"]) ? maze_from_file(args["maze"]) : PyTrade().Trade(env_config)
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
        sc = SeedCache(maxsize=round(Int, args["num-elites"]*1.2))

    end

    pop = [[rand(UInt32)] for _ in 1:pop_size]
    best = (-Inf, [])
    archive = Set()
    BC = nothing
    F = nothing

    # ###############
    # load checkpoint
    # ###############
    check_name = "outs/$clsname/$expname/check.jld2"
    met_csv_name = "outs/$clsname/$expname/metrics.csv"
    start_gen = 1
    # check if check exists on the file system
    if isfile(check_name)
        df = isfile(met_csv_name) ? CSV.read(met_csv_name, DataFrame) : nothing
        check = load(check_name)
        start_gen = check["gen"] + 1
        F, BC, best, archive = getindex.((check,), ["F", "BC", "best","archive"])
        pop = create_next_pop(start_gen, check["pop"], args["num-elites"])
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

        llog(islocal=args["local"], name=logname) do logfile
            ts(logfile, "computing elite by re-evaluating top performers")
        end
        @assert length(F) == length(BC) == pop_size
        elite = compute_elite(fitness, pop, F, k=args["num-elites"], n=2)

        if elite[1] > best[1]
            llog(islocal=args["local"], name=logname) do logfile
                ts(logfile, "New best ind found, F=$(elite[1])")
            end
            best = elite
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
        # println("printing pop and novelties")
        # for i in 1:10
        #     println(round(novelties[i], digits=2),"--", pop[i])
        # end
        @assert length(novelties) == pop_size

        reorder!(novelties, F, BC, pop)

        # LOG
        if g % 1 == 0
            ts("log start")
            models = Dict("f0a0" => re(reconstruct(sc, mi, best[2], args["mutation-rate"])),
            "f1a0" => re(reconstruct(sc, mi, best[2], args["mutation-rate"])))

            # Compute and write metrics
            outdir = "outs/$clsname/$expname/$g"
            run(`mkdir -p $outdir`)
            rew_dict, mets, _ = run_batch(env, models, args, evaluation=true, render_str=outdir)
            df = update_df(df, mets)
            write_mets(met_csv_name, df)

            # Log to file
            avg_self_fit = round((rew_dict["f0a0"] + rew_dict["f1a0"]) / 2; digits=2)
            llog(islocal=args["local"], name=logname) do logfile
                ts(logfile, "Generation $g: $avg_self_fit")
            end
            plot_bcs(outdir, env, BC, novelties)

            # Save checkpoint
            !args["local"] && save(check_name, Dict("gen"=>g, "pop"=>pop, "archive"=>archive, "BC"=> BC, "F"=>F, "best"=>best))
            ts("log end")
        end
        # TODO: CHANGE IF GA EVER WORKS
        if best[1] > 0 
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
        @everywhere cache_elites!(sc, mi, $pop[1:args["num-elites"]], args["mutation-rate"])
        pop = create_next_pop(g, pop, args["num-elites"])
    end
end
