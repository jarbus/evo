root_dir = dirname(@__FILE__)  |> dirname |> String
@testset "test_rollout_maze" begin
    println("test_rollout")
    env = maze_from_file("$root_dir/mazes/test_maze.txt")
    batch_size = 1
    θ, re = make_model(:large,
            (env.obs_size..., batch_size),
            env.num_actions,
            vbn=false,
            lstm=true) |> Flux.destructure


    models = Dict("f0a0"=>re(reconstruct(UInt32.([3, 4, 5]), length(θ))))
    args = Dict("episode-length"=>400, "batch-size"=>batch_size)
    rewards, mets, bc = run_batch(env, models, args)
    @test mets isa Nothing
end
