using CSV
using Distributed
using DataFrames
using FileIO
using Infiltrator
using Logging


@everywhere begin
    using EvoTrade
    args = $args
    expname = args["exp-name"]
    clsname = args["cls-name"]
    using Flux
    using Statistics
    using StableRNGs

    mets_to_return = vcat([
                    ["$(met)_$f" for f in 0:args["food-types"]-1 for met in
                          ("exchange", "picks", "places", "rew_light")]]...,
                          ["rew_base_health"], ["rew_acts"], ["mut_exchanges"], ["gives"], ["takes"])
    env_config = mk_env_config(args)

    function fitness(group::Vector, eval_gen)
        counts = Dict{Int, Int}()
        for (i, _) in group
            counts[i] = get(counts, i, 0) + 1
        end
        group = decompress_group(group, prefixes)
        # creates mapping of pid_copy to params
        params = Dict(aid(i, c)=>reconstruct(sc, mi, seeds, e_idxs) for (i, seeds, e_idxs) in group for c in 1:counts[i])
        models = Dict(aid=>re(param) for (aid, param) in params)
        rew_dict, mets, bc_dict, info_dict = run_batch(env, models, args, evaluation=true)
        rews, bcs, infos = Dict(), Dict(), Dict{Any, Any}("avg_walks"=>Dict())
        for (i, _) in group 
            i < 0 && continue # skip elites
            a_id = aid(i, 1)
            rews[i] = [rew_dict[a_id]]
            bcs[i] = [bc_dict[a_id]]
            infos["avg_walks"][i] = [info_dict["avg_walks"][a_id]]
            for c in 2:counts[i]
                a_id = aid(i, c)
                push!(rews[i], rew_dict[a_id])
                push!(bcs[i], bc_dict[a_id])
                push!(infos["avg_walks"][i], info_dict["avg_walks"][a_id])
            end
            if !eval_gen
                infos["avg_walks"][i] = []
            end
        end
        for (i, _) in group
            i < 0 && continue
            @assert i in keys(rews)
        end
        infos["min_params"] = minimum(minimum.(values(params)))
        infos["max_params"] = maximum(maximum.(values(params)))
        infos["mets"] = eval_gen ? filter(p->p.first in mets_to_return, mets) : []
        rews, bcs, infos
    end
end

function main()
    dt_str = args["datime"]
    logname="runs/$clsname/$dt_str-$expname.log"
    check_name = "outs/$clsname/$expname/check.jld2"
    met_csv_name = "outs/$clsname/$expname/metrics.csv"
    sc_name = "outs/$clsname/$expname/sc.jld2"
    start_gen = 1
    global_logger(EvoTradeLogger(args["local"] ? stdout : logname))
    df = nothing
    wp = WorkerPool(workers())
    pop_size = args["pop-size"]
    pop = [Vector{Float32}([rand(UInt32)]) for _ in 1:pop_size]
    elites = Vector{Dict}()
    best = (-Inf, [])
    archive = Set()
    BC = nothing
    F = nothing
    γ = args["exploration-rate"]
    global metrics

    @info "cls: $clsname"
    @info "exp: $expname"
    @info "Running on commit: "*read(`git rev-parse --short HEAD`, String)
    
    @info "Initializing on all workers"
    @everywhere begin
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
        global sc = SeedCache(maxsize=args["num-elites"]*3)
        prefixes = Dict()
    end
    @info "model has $model_size params"

    if isfile(check_name)
        @info "Loading from checkpoints"
        df = isfile(met_csv_name) ? CSV.read(met_csv_name, DataFrame) : nothing
        # We can get pre-empted while checkpointing, meaning the latest check
        # might be invalid. We always mv the previous jld2 files to *.jld2-old,
        # and boot from that if the latest check is corrupt.
        try
            global check = load(check_name)
            global sc = load(sc_name)["sc"]
        catch
            global check = load(check_name*"-old")
            global sc = load(sc_name*"-old")["sc"]
        end
        start_gen = check["gen"] + 1
        γ = check["gamma"]
        F, BC, best, archive, novelties = getindex.((check,), ["F", "BC", "best","archive", "novelties"])
        global pop = check["pop"]
        global elites = check["elites"]

        cache_elites!(sc, mi, elites)

        @info "resuming from gen $start_gen"
    end

    for g in start_gen:args["num-gens"]
        eval_gen = g % 50 == 1
        global prefixes
        @info "starting generation $g"
        @info "creating groups"

        groups = create_rollout_groups(pop, elites, args["rollout-group-size"], args["rollout-groups-per-mut"])
        groups = add_elite_idxs_to_groups(groups, elites)
        avg_set_len = 0
        avg_pop_lens = 0
        num_inds = 0
        union_set = Set()
        for g in groups
            for ind in g
                eset = ind[3]
                avg_set_len += length(eset)
                num_inds += 1
                avg_pop_lens += length(ind[2])
                union_set = union(union_set, eset)
            end
        end
        @info "avg num_elites_cached len: $(avg_set_len/num_inds)"
        @info "union set: $(union_set)"
        @info "avg pop len: $(avg_pop_lens/num_inds)"
        @info "compressing groups"
        groups = compress_groups(groups, prefixes)

        @info "pmapping"
        fetches = pmap(wp, groups) do g
            fitness(g, eval_gen)
        end

        F = [[] for _ in 1:pop_size]
        BC = [[] for _ in 1:pop_size]
        walks_list = [[] for _ in 1:pop_size]
        group_rollout_metrics = []
        for fet in fetches, idx in keys(fet[1])
            push!(F[idx], fet[1][idx]...)
            push!(BC[idx], fet[2][idx]...)
            push!(walks_list[idx], fet[3]["avg_walks"][idx]...)
            push!(group_rollout_metrics, fet[3]["mets"])
        end
        @assert all(length.(F) .>= args["rollout-groups-per-mut"])
        # TODO: change these to maximums
        F = [maximum(f) for f in F]
        BC = [max_bc(bcs) for bcs in BC]
        if eval_gen
            walks = [average_walk(w) for w in walks_list]
            rollout_metrics = mergewith(vcat, group_rollout_metrics...)
        end


        @assert length(F) == length(BC) == pop_size
        elite = (maximum(F), pop[argmax(F)])

        # update elite and modify exploration rate
        if elite[1] > best[1]
            γ = 0.1
            @info "New best ind found, F=$(elite[1]), γ decreased to $γ"
            best = elite
        else
            γ = clamp(γ + 0.02, 0, 0.9)
            @info "no better elite found, set γ to $γ"
        end
        
        add_to_archive!(archive, BC, pop, args["archive-prob"])

        bc_matrix = hcat(BC...)
        pop_and_arch = hcat([bc for (bc, _) in archive]..., bc_matrix)
        @assert size(pop_and_arch, 2) == pop_size + length(archive)
        @assert size(bc_matrix, 2) == pop_size
        @assert size(bc_matrix, 1) == size(BC[1], 1)

        @info "computing novelties"
        novelties = compute_novelties(bc_matrix, pop_and_arch, k=min(pop_size-1, 25))
        @assert length(novelties) == pop_size
        @info "most novel bc: $(BC[argmax(novelties)])"
        @info "most fit bc: $(BC[argmax(F)]), fitness $(maximum(F))"

        # Run evaluation, collect statistics, and checkpoint every 50 gens
        if eval_gen
            prefixes = compute_prefixes(elites)
            @info "distributing prefixes: $(prefixes)"
            @everywhere prefixes = $prefixes

            @spawnat 1 begin
               @info "log start"
               # Compute and write metrics
               outdir = "outs/$clsname/$expname/"*string(g, pad=3, base=10)

               run(`mkdir -p $outdir`)

               plot_grid_and_walks(env, "$outdir/pop.png", grid, walks, novelties, F, args["num-elites"], γ)

               eval_best_idxs = sortperm(F, rev=true)[1:ceil(Int, args["num-elites"]*γ)]
               @info "evaluating inds with fitnesses $(F[eval_best_idxs])"
               eval_group_idxs = [rand(eval_best_idxs, args["rollout-group-size"]) for _ in 1:1]
               eval_group_seeds = [[pop[idx] for idx in idxs] for idxs in eval_group_idxs]
               eval_metrics_vec = pmap(wp, eval_group_seeds) do group_seeds
                  models = Dict("p$i" => re(reconstruct(sc, mi, seeds)) for (i, seeds) in enumerate(group_seeds))
                  str_name = joinpath(outdir, string(hash(group_seeds))*"-"*string(myid()))
                  _, metrics, _, _ = run_batch(env, models, args, evaluation=true, render_str=str_name, batch_size=1)
                  metrics
               end
               isnothing(args["maze"]) && vis_outs(outdir, args["local"])

               metrics_csv = Dict()
               @info "Logging evaluation metrics"
               eval_metrics = mergewith(vcat, eval_metrics_vec...)
               @info "eval metrics: $(eval_metrics)"
               for (met_name, met_vec) in eval_metrics
                   log_mmm!(metrics_csv, "eval_"*met_name, met_vec)
               end
               global rollout_metrics
               @info "Logging population metrics"
               @info "rollout metrics: $(rollout_metrics)"
               for (met_name, met_vec) in rollout_metrics
                   log_mmm!(metrics_csv, "pop_"*met_name, met_vec)
               end
               muts = g > 1 ? [mr(pop[i]) for i in 1:pop_size] : [0.0]
               metrics_csv["gamma"] = γ
               metrics_csv["min_param"] = minimum([fet[3]["min_params"] for fet in fetches])
               metrics_csv["max_param"] = maximum([fet[3]["max_params"] for fet in fetches])
               log_mmm!(metrics_csv, "mutation_rate", muts)
               log_mmm!(metrics_csv, "fitness", F)
               log_mmm!(metrics_csv, "novelty", novelties)
               df = update_df(df, metrics_csv)
               write_mets(met_csv_name, df)

               # Save checkpoint
               @info "savefile"
               isfile(check_name) && run(`mv $check_name $check_name-old`)
               save(check_name, Dict("gen"=>g, "gamma"=>γ, "pop"=>pop, "archive"=>archive, "BC"=> BC, "F"=>F, "best"=>best, "novelties"=>novelties, "elites"=>elites))

               # Save seed cache without parameters
               isfile(sc_name) && run(`mv $sc_name $sc_name-old`)
               sc_no_params = SeedCache(maxsize=3*args["num-elites"])
               for (k,v) in sc
                   sc_no_params[k] = Dict(ke=>ve for (ke,ve) in v if ke != :params)
               end
               save(sc_name, Dict("sc"=>sc_no_params))
           end
        end

        if best[1] > 15
            best_bc = BC[argmax(F)]
            @info "Returning: Best individal found with fitness $(best[1]) and BC $best_bc"
            save("outs/$clsname/$expname/best.jld2", Dict("best"=>best, "bc"=>best_bc))
            return
        end

        pop, elites = create_next_pop(g, sc, pop, F, novelties, BC, γ, args["num-elites"])
        @info "cache_elites"
        cache_elites!(sc, mi, elites)
    end
end
