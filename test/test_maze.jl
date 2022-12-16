using EvoTrade
using Test
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
    @test sum(obs.==1) == 4 # check that each of 4 frames has a player pos
    @test ndims(obs) == 4
    r, done = step!(env, up)
    @test env.locations[4] != env.locations[3]
    obs = get_obs(env)
    @test obs[env.locations[4]..., 4] == obs[env.locations[3]..., 3] == 1
    @test obs[env.locations[3]..., 4] != obs[env.locations[3]..., 3]
    @test sum(obs.==1) == 4
end
@testset "test_plot_bcs" begin
    test_dir = joinpath(root_dir, "test/test")
    run(`mkdir -p $test_dir`, wait=false)
    maze_png = joinpath(test_dir, "maze.png")
    run(`rm $maze_png`, wait=false)
    sleep(0.1)
    env = maze_from_file("$root_dir/mazes/test_maze.txt")
    plot_bcs(test_dir, env, [(2, 2), (2, 3), (3,4)])
    @test isfile(maze_png)
    run(`rm $maze_png`, wait=false)
    sleep(0.1)
    env = maze_from_file("$root_dir/mazes/hard-maze.txt")
    plot_bcs(test_dir, env, [(2, 2), (2, 3), (3,4)])
    @test isfile(maze_png)
    run(`rm $maze_png`, wait=false)
    env = maze_from_file("$root_dir/mazes/hard-maze.txt")
    plot_bcs(test_dir, env, [(2, 2), (2, 3), (3,4)], [500,3,2])
    @test isfile(maze_png)
    run(`rm $maze_png`, wait=false)
end
