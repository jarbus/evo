using EvoTrade
using Flux
using Test
root_dir = dirname(@__FILE__)  |> dirname |> String
@testset "test_rollout_maze" begin
    env = maze_from_file("$root_dir/mazes/hard-maze.txt")
    batch_size = 1
    θ, re = make_model(:large,
            (env.obs_size..., batch_size),
            env.num_actions,
            vbn=false,
            lstm=true) |> Flux.destructure

    nt = NoiseTable(StableRNG(123), length(θ), 1, 1f0)
    models = Dict("f0a0"=>re(reconstruct(nt, UInt32.([3, 4, 5]))))
    args = Dict("episode-length"=>400, "batch-size"=>batch_size)
    rewards, mets, bc = run_batch(env, models, args)
    @test mets isa Nothing
end
