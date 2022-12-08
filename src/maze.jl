module Maze

export maze_from_file, step!, reset!, sample_batch, get_obs, MazeEnv, print_maze
using StatsBase
using Plots

mutable struct MazeEnv
    grid::Array{Int64,2}
    start::Tuple{Int64,Int64}
    locations::Vector{Tuple{Int64,Int64}}
    goals::Vector{Tuple{Int64,Int64}}
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
    grid = hcat(grid[end:-1:1]...)
    @assert 2 in grid
    @assert 3 in grid
    # find the start location
    start_pos = findfirst(grid .== 2) |> Tuple
    end_pos = findall(grid .== 3) .|> Tuple
    # create the environment
    return MazeEnv(grid, start_pos, [start_pos for _ in 1:4], end_pos, (size(grid)...,4),4)
end

function get_obs(env::MazeEnv)
    frames = cat([env.grid ./ 4f0 for _ in 1:4]..., dims=3) 
    @assert size(frames) == env.obs_size
    for i in 1:4
        frames[env.locations[i]..., i] = 1.0
    end
    frames[:,:,:,:]
end

function sample_batch(probs::Matrix{Float32})
  [sample(1:size(probs, 1), Weights(probs[:, i])) for i in 1:size(probs, 2)]
end

function reset!(env::MazeEnv)
    env.locations = [env.start for _ in 1:4]
end
    

function step!(env::MazeEnv, act::Int64)
    # act is 1: up, 2: right, 3: down, 4: left
    # get the current location
    r, c = env.locations[4]
    # otherwise, move the agent
    acts = [(r-1, c) (r, c+1) (r+1, c) (r, c-1)]
    new_pos = acts[act]
    if env.grid[new_pos...] != 1
        push!(env.locations, new_pos)
        popfirst!(env.locations)
    end
    # clip location at grid borders
    env.locations[4] = min.(max.(env.locations[4], 1), size(env.grid))
    # get the current grid value
    if env.grid[env.locations[4]...] == 3
        return 10, true
    end

    return -dist(env), false
end

function dist(env::MazeEnv)
    # compute euclidean distance between env.location and env.goal
    return min([sqrt(sum((env.locations[4] .- g).^2)) for g in env.goals]...)
end

function print_maze(env::MazeEnv)
    # print the grid
    for j in size(env.grid, 1):-1:1
        for i in 1:size(env.grid, 2)
            if (i, j) == env.locations[4]
                print("A")
            elseif (i, j) == env.start
                print("S")
            elseif (i, j) in env.goals
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


end
