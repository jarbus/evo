using EvoTrade
using Flux
using Test
root_dir = dirname(@__FILE__)  |> dirname |> String
# @testset "test_rollout_maze" begin
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


@testset "test_rollout_trade" begin
    expname = ["--exp-name", "test", "--cls-name","test", "--local", "--datime", "test"]
    arg_vector = read("$root_dir/afiles/cls-test/test-ga-trade.arg", String) |> split
    args = parse_args(vcat(arg_vector, expname), get_arg_table())
    env_config = mk_env_config(args)
    env = PyTrade().Trade(env_config)

    θ, re = make_model(:large,
            (env.obs_size..., 2),
            env.num_actions,
            vbn=false,
            lstm=true) |> Flux.destructure
    models = Dict("f0a0"=>re(θ), "f1a0"=>re(θ))
    rew, met, bc, info = run_batch(env_config, models,args)
    @test length(info["avg_walks"]["f0a0"]) == args["episode-length"] * 2
    @test length(info["avg_walks"]["f1a0"]) == args["episode-length"] * 2
end


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
