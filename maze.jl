module Maze

export maze_from_file, step!, reset!, test_maze

mutable struct MazeEnv
    grid::Array{Int64,2}
    start::Tuple{Int64,Int64}
    location::Tuple{Int64,Int64}
    goal::Tuple{Int64,Int64}
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
    return MazeEnv(grid, start_pos, start_pos, end_pos)
end

function reset!(env::MazeEnv)
    env.location = env.start
end
    

function step!(env::MazeEnv, act::Int64)
    # act is 0: up, 1: right, 2: down, 3: left
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
        return 10.0, true
    end

    return -1.0, false
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

    env = maze_from_file("test_maze.txt")
    reset!(env)
    for i in 1:10
        print_maze(env)
        act = rand(1:4)
        println("Act: ", act)
        r, done = step!(env, act)
        println("Player position: $(env.location)")
        println("Reward: $r")
    end
end

end
