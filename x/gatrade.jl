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

    function fitness(group::Vector)
        counts = Dict{Int, Int}()
        for (i, _) in group
            counts[i] = get(counts, i, 0) + 1
        end
        # creates mapping of pid_copy to params
        params = Dict(aid(i, c)=>reconstruct(sc, mi, seeds) for (i, seeds) in group for c in 1:counts[i])
        models = Dict(aid=>re(param) for (aid, param) in params)
        rew_dict, _, bc_dict, info_dict = run_batch(env, models, args, evaluation=true)
        rews, bcs, infos = Dict(), Dict(), Dict{Any, Any}("avg_walks"=>Dict())
        for (i, _) in group 
            i < 0 && continue # skip elites
            a_id = aid(i, 1)
            rews[i] = rew_dict[a_id]
            bcs[i] = bc_dict[a_id]
            infos["avg_walks"][i] = [info_dict["avg_walks"][a_id]]
            for c in 2:counts[i]
                a_id = aid(i, c)
                rews[i] += rew_dict[a_id]
                bcs[i] .+= bc_dict[a_id]
                push!(infos["avg_walks"][i], info_dict["avg_walks"][a_id])
            end
            rews[i] /= counts[i]
            bcs[i] ./= counts[i]
        end
        infos["min_params"] = minimum(minimum.(values(params)))
        infos["max_params"] = maximum(maximum.(values(params)))
        rews, bcs, infos
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
            env = maze_from_file(args["maze"])
            grid = env.grid 
        else
            env = PyTrade().Trade(env_config)
            EvoTrade.Trade.reset!(env)
            grid = env.table 
        end

        m = make_model(Symbol(args["model"]),
            (env.obs_size..., args["batch-size"]),
            env.num_actions,
            vbn=args["algo"]=="es",
            lstm=args["lstm"])
        θ, re = Flux.destructure(m)
        mi = ModelInfo(m)
        model_size = length(θ)
        # pass mazeenv struct or trade config dict
        env = env isa MazeEnv ? env : env_config
        global sc = SeedCache(maxsize=args["num-elites"]*2)
    end
    llog(islocal=args["local"], name=logname) do logfile
        ts(logfile, "model has $model_size params")
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
        global pop = check["pop"]
        global elites = check["elites"]

        for p in pop
            @assert elite(p) in keys(sc)
        end
        #pop, elites = create_next_pop(start_gen-1, sc, check["pop"], F, novelties, BC, γ, args["num-elites"])
        #@everywhere cache_elites!(sc, mi, $elites)
        ts("resuming from gen $start_gen")
    end

    for g in start_gen:args["num-gens"]

        llog(islocal=args["local"], name=logname) do logfile
            ts(logfile, "pmapping")
        end

        if g == 1
            groups = create_rollout_groups(pop, args["rollout-group-size"], pop_size)
        else
            groups = create_rollout_groups(pop, elites, args["rollout-group-size"], pop_size)
        end

        fetches = pmap(groups) do g
            fitness(g)
        end

        F = [[] for _ in 1:pop_size]
        BC = [[] for _ in 1:pop_size]
        walks_list = [[] for _ in 1:pop_size]
        for fet in fetches, idx in keys(fet[1])
            push!(F[idx], fet[1][idx])
            push!(BC[idx], fet[2][idx])
            push!(walks_list[idx], fet[3]["avg_walks"][idx]...)
        end
        F = [mean(f) for f in F]
        BC = [average_bc(bcs) for bcs in BC]
        walks = [average_walk(w) for w in walks_list]


        @assert length(F) == length(BC) == pop_size
        elite = (maximum(F), pop[argmax(F)])

        # update elite and modify exploration rate
        if elite[1] > best[1]
            γ = clamp(γ / 2, 0, 0.9)
            llog(islocal=args["local"], name=logname) do logfile
                ts(logfile, "New best ind found, F=$(elite[1]), γ decreased to $γ")
            end
            best = elite
        else
            γ = clamp(γ + 0.05, 0, 0.9)
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

            # Compute and write metrics
            outdir = "outs/$clsname/$expname/"*string(g, pad=3, base=10)

            run(`mkdir -p $outdir`)

            plot_grid_and_walks(env, "$outdir/pop.png", grid, walks, novelties, F)

            # TODO make this select random rollouts with duplicates of fit agents
            # run parallel visualization on most fit most novel members 
            eval_best_idxs = sortperm(F, rev=true)[1:args["rollout-group-size"]]
            eval_group_idxs = [rand(eval_best_idxs, args["rollout-group-size"]) for _ in 1:10]
            eval_group_seeds = [[pop[idx] for idx in idxs] for idxs in eval_group_idxs]
            mets = pmap(eval_group_seeds) do group_seeds
                models = Dict("p$i" => re(reconstruct(sc, mi, seeds)) for (i, seeds) in enumerate(group_seeds))
                str_name = joinpath(outdir, string(hash(group_seeds))*"-"*string(myid()))
                rew_dict, metrics, _, _ = run_batch(env, models, args, evaluation=true, render_str=str_name, batch_size=1)
                metrics
            end
            # average all the rollout metrics
            mets = Dict(k => mean([m[k] for m in mets]) for k in keys(mets[1]))
            isnothing(args["maze"]) && vis_outs(outdir, args["local"])

            muts = g > 1 ? [mr(pop[i]) for i in 1:pop_size] : [0.0]
            mets["gamma"] = γ
            mets["min_param"] = minimum([fet[3]["min_params"] for fet in fetches])
            mets["max_param"] = maximum([fet[3]["max_params"] for fet in fetches])
            log_mmm!(mets, "mutation_rate", muts)
            log_mmm!(mets, "fitness", F)
            log_mmm!(mets, "novelty", novelties)
            df = update_df(df, mets)
            write_mets(met_csv_name, df)

            # Log to file
            avg_self_fit = round(mets["fitness_mean"]; digits=2)
            llog(islocal=args["local"], name=logname) do logfile
                ts(logfile, "Generation $g: $avg_self_fit")
            end

            llog(islocal=args["local"], name=logname) do logfile
                ts(logfile, "log end")
            end
        end
        if best[1] > 15
            best_bc = BC[argmax(F)]
            llog(islocal=args["local"], name=logname) do logfile
                ts("Returning: Best individal found with fitness $(best[1]) and BC $best_bc")
            end
            save("outs/$clsname/$expname/best.jld2", Dict("best"=>best, "bc"=>best_bc))
            return
        end

        pop, elites = create_next_pop(g, sc, pop, F, novelties, BC, γ, args["num-elites"])
        llog(islocal=args["local"], name=logname) do logfile
            ts(logfile, "cache_elites")
        end
        # Save checkpoint
        save(check_name, Dict("gen"=>g, "gamma"=>γ, "pop"=>pop, "archive"=>archive, "BC"=> BC, "F"=>F, "best"=>best, "novelties"=>novelties, "elites"=>elites))

        @everywhere cache_elites!(sc, mi, $elites)

        # Save seed cache without parameters
        sc_no_params = SeedCache(maxsize=2*args["num-elites"])
        for (k,v) in sc
            sc_no_params[k] = Dict(ke=>ve for (ke,ve) in v if ke != :params)
        end
        save(sc_name, Dict("sc"=>sc_no_params))
    end

end
