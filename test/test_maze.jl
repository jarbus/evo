
left, up, right, down = 1, 2, 3, 4
root_dir = dirname(@__FILE__)  |> dirname |> String
@testset "test_maze" begin
    env = maze_from_file("$root_dir/mazes/test_maze.txt")
    @assert length(env.locations) == 4
    reset!(env)
    solution = [up, up, right, right, right, right, down, down]
    rews = [-sqrt(2), -1, 10, -1, -1, -1, -1, 10]
    for i in 1:8
        act = solution[i]
        r, done = step!(env, act)
        @test r == rews[i] != i != env.locations[4]
    end
end
@testset "test_maze_obs" begin
    env = maze_from_file("$root_dir/mazes/test_maze.txt")
    reset!(env)
    obs = get_obs(env)
    @test sum(obs.==1) == 4
    @test ndims(obs) == 4
    r, done = step!(env, up)
    @test env.locations[4] != env.locations[3]
    obs = get_obs(env)
    @test obs[env.locations[4]..., 4] == obs[env.locations[3]..., 3] == 1
    @test obs[env.locations[3]..., 4] != obs[env.locations[3]..., 3]
    @test sum(obs.==1) == 4
end
function main()
@testset "test_plot_bcs" begin
    run(`rm $root_dir/outs/test/maze.png`, wait=false)
    sleep(0.1)
    env = maze_from_file("$root_dir/mazes/test_maze.txt")
    plot_bcs("test", env, [(2, 2), (2, 3), (3,4)])
    @test isfile("outs/test/maze.png")
end
end
