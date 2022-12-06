using CSV
using Distributed
using DataFrames
using FileIO
using Infiltrator

@everywhere begin
    using EvoTrade
    args = $args
    expname = args["exp-name"]
    using Flux
    using Statistics
    using StableRNGs

    env_config = mk_env_config(args)

    function fitness(p1::T, p2::T) where T<:Vector{<:UInt32}
        models = Dict("f0a0" => re(reconstruct(p1, model_size, args["mutation-rate"])),
        "f1a0" => re(reconstruct(p2, model_size, args["mutation-rate"])))
        rew_dict, _, bc = run_batch(env, models, args)
        rew_dict["f0a0"], rew_dict["f1a0"], bc["f0a0"], bc["f1a0"]
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
        # pass mazeenv struct or trade config dict
        env = env isa MazeEnv ? env : env_config

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
        F, BC, best, archive = getindex.((check,), ["F", "BC", "best","archive"])
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
        elite = compute_elite(fitness, pop, F, k=args["num-elites"], n=2)

        if elite[1] > best[1]
            llog(islocal=args["local"], name=logname) do logfile
                ts(logfile, "New best ind found, F=$(elite[1])")
            end
            best = elite
        end
        
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
            rew_dict, mets, _ = run_batch(env, models, args, evaluation=false, render_str=outdir)
            df = update_df(df, mets)
            write_mets(met_csv_name, df)

            # Log to file
            avg_self_fit = round((rew_dict["f0a0"] + rew_dict["f1a0"]) / 2; digits=2)
            llog(islocal=args["local"], name=logname) do logfile
                ts(logfile, "Generation $g: $avg_self_fit")
            end

            # Save checkpoint
            !args["local"] && save(check_name, Dict("gen"=>g, "pop"=>pop, "archive"=>archive, "BC"=> BC, "F"=>F, "best"=>best))

            # TODO: CHANGE IF GA EVER WORKS
            if best[1] > 0 
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
