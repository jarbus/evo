using EvoTrade
using Test

function main()

@testset "test_maze" begin
    env = maze_from_file("mazes/test_maze.txt")
    reset!(env)
    solution = [4, 4, 3, 3, 3, 3, 2, 2]
    rews = [-sqrt(2), -1, 10, -1, -1, -1, -1, 10]
    for i in 1:8
        act = solution[i]
        r, done = step!(env, act)
        @test r == rews[i]
    end
end
end

