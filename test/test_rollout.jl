using EvoTrade
using Flux
using Test
root_dir = dirname(@__FILE__)  |> dirname |> String
@testset "test_rollout_maze" begin
    env = maze_from_file("$root_dir/mazes/hard-maze.txt")
    batch_size = 1
    m = make_model(:large,
            (env.obs_size..., batch_size),
            env.num_actions,
            vbn=false,
            lstm=true)
    θ, re = Flux.destructure(m)
    mi = ModelInfo(m)
    sc = SeedCache(maxsize=10)
    params = reconstruct(sc, mi, [1.0])
    models = Dict("p1"=>re(params))
    args = Dict("episode-length"=>400, "batch-size"=>batch_size)
    rewards, mets, bc, infos = run_batch(env, models, args, evaluation=true)
    walks1 = infos["avg_walks"]["p1"]
    rewards, mets, bc, infos = run_batch(env, models, args, evaluation=true)
    walks2 = infos["avg_walks"]["p1"]
    # test that a rollout with the same model is the same
    @test walks1 == walks2

    # the following segments of code are designed to assert that identical seeds have
    # identical rollouts in a stripped down version of x/gatrade.jl
    pop_size = 10
    γ=0.5
    n_elites = 4

    pop = [rand(1.0:1000.0, 1) for _ in 1:pop_size]
    elites = nothing
    for g in 1:2
        fets = map(pop) do seeds
            e_params = nothing
            e_walks = nothing
            if !isnothing(elites)
                for e in elites
                    if seeds == e[:seeds]
                        e_params = e[:params]
                        e_walks = e[:walk]
                        break
                    end
                end
            end
            params = reconstruct(sc, mi, seeds)
            models = Dict("p1"=>re(params))
            args = Dict("episode-length"=>40, "batch-size"=>1)
            rewards, mets, bc, infos = run_batch(env, models, args, evaluation=true)

            if !isnothing(e_params)
                @test params == e_params
                @test infos["avg_walks"]["p1"] == e_walks
                @test length(infos["avg_walks"]["p1"]) == length(e_walks)
            end
            rewards["p1"], nothing, bc["p1"], infos
        end

        F = [f[1] for f in fets]
        BC = [f[3] for f in fets]
        W = [f[4]["avg_walks"]["p1"] for f in fets]
        # assert that all elements of fobs are the same

        if !isnothing(elites)
            for e in elites
                idx = findfirst(x->x==e[:seeds], pop)
                walk = W[idx]
                @test e[:params] == reconstruct(sc, mi, pop[idx])
                @test idx <= n_elites
                @test length(e[:walk]) == length(walk)
                @test e[:walk] == walk
            end
        end

        bc_matrix = hcat(BC...)
        novelties = compute_novelties(bc_matrix, bc_matrix, k=min(pop_size-1, 25))
        new_pop, elites = create_next_pop(1, sc, pop, F, novelties, BC, γ, n_elites)
        for e in elites
            e[:walk] = W[findfirst(x->x==e[:seeds], pop)]
        end
        pop = new_pop

        cache_elites!(sc, mi, elites)
        p1_2 = reconstruct(sc, mi, pop[1])
    end

end

# @testset "test_rollout_trade" begin
#     expname = ["--exp-name", "test", "--cls-name","test", "--local", "--datime", "test"]
#     arg_vector = read("$root_dir/afiles/cls-test/test-ga-trade.arg", String) |> split
#     args = parse_args(vcat(arg_vector, expname), get_arg_table())
#     env_config = mk_env_config(args)
#     env = PyTrade().Trade(env_config)
#
#     θ, re = make_model(:large,
#             (env.obs_size..., 2),
#             env.num_actions,
#             vbn=false,
#             lstm=true) |> Flux.destructure
#     models = Dict("f0a0"=>re(θ), "f1a0"=>re(θ))
#     rew, met, bc, info = run_batch(env_config, models,args)
#     @test length(info["avg_walks"]["f0a0"]) == args["episode-length"] / 2
#     @test length(info["avg_walks"]["f1a0"]) == args["episode-length"] / 2
# end


# @testset "profile_rollout_maze" begin
#     env = maze_from_file("$root_dir/mazes/hard-maze.txt")
#     batch_size = 1
#     θ, re = make_model(:large,
#             (env.obs_size..., batch_size),
#             env.num_actions,
#             vbn=false,
#             lstm=true) |> Flux.destructure

#     nt = NoiseTable(StableRNG(123), length(θ), 1, 1f0)
#     models = Dict("f0a0"=>re(reconstruct(nt, UInt32.([3, 4, 5]))))
#     args = Dict("episode-length"=>400, "batch-size"=>batch_size)
#     rewards, mets, bc = run_batch(env, models, args)
#     @test mets isa Nothing
# end
