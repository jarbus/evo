module Maze

export maze_from_file, step!, reset!, test_maze, sample_batch, get_obs
using StatsBase

mutable struct MazeEnv
    grid::Array{Int64,2}
    start::Tuple{Int64,Int64}
    location::Tuple{Int64,Int64}
    goal::Tuple{Int64,Int64}
    obs_size::Tuple{Int64,Int64,Int64}
    num_actions::Int64
end

function maze_from_file(name::String)
    # read in the maze from a file
    # 1 is a wall, 0 is a passable space
    # 2 is the start, 3 is the goal
    grid = []
    open(name) do f
        for line in eachline(f)
            push!(grid, [parse(Int64, x) for x in split(line,"")])
        end
    end
    # cast grid to matrix
    grid = hcat(grid...)
    @assert 2 in grid
    @assert 3 in grid
    # find the start location
    start_pos = findfirst(grid .== 2) |> Tuple
    end_pos = findfirst(grid .== 3) |> Tuple
    # create the environment
    return MazeEnv(grid, start_pos, start_pos, end_pos, (size(grid)...,1),4)
end

function get_obs(env::MazeEnv)
    return env.grid[:,:,:,:] ./ 3f0
end

function sample_batch(probs::Matrix{Float32})
  [sample(1:size(probs, 1), Weights(probs[:, i])) for i in 1:size(probs, 2)]
end

function reset!(env::MazeEnv)
    env.location = env.start
end
    

function step!(env::MazeEnv, act::Int64)
    # act is 1: up, 2: right, 3: down, 4: left
    # get the current location
    r, c = env.location
    # otherwise, move the agent
    acts = [(r-1, c) (r, c+1) (r+1, c) (r, c-1)]
    new_pos = acts[act]
    if env.grid[new_pos...] != 1
        env.location = new_pos
    end
        # clip location at grid borders
    env.location = min.(max.(env.location, 1), size(env.grid))
    # get the current grid value
    if env.grid[env.location...] == 3
        return 10, true
    end

    return -dist(env), false
end

function dist(env::MazeEnv)
    # compute euclidean distance between env.location and env.goal
    return sqrt(sum((env.location .- env.goal).^2))
end

function print_maze(env::MazeEnv)
    # print the grid
    for i in 1:size(env.grid, 1)
        for j in 1:size(env.grid, 2)
            if (i, j) == env.location
                print("A")
            elseif (i, j) == env.start
                print("S")
            elseif (i, j) == env.goal
                print("G")
            elseif env.grid[i, j] == 1
                print("#")
            else
                print(" ")
            end
        end
        println()
    end
end


function test_maze()

    env = maze_from_file("mazes/test_maze.txt")
    reset!(env)
    solution = [4, 4, 3, 3, 3, 3, 2, 2]
    for i in 1:8
        print_maze(env)
        act = solution[i]
        println("Act: ", act)
        r, done = step!(env, act)
        println("Player position: $(env.location)")
        println("Reward: $r")
    end
end

end
