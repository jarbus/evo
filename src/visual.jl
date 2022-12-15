function plot_bcs(dirname::String, env::MazeEnv, bcs::Vector, novs::Vector=[])
    @assert length(bcs) == length(novs) || length(novs) == 0
    maze_matrix = env.grid'
    poses = [(pos[1], pos[2]) for pos in bcs]
    colors = :blue
    if length(novs) > 0
        max_nov = max(maximum(novs), 0.1)
        colors = [colorant"blue"*0.8 + colorant"yellow" * nov/max_nov for nov in novs]
    end
    hm = heatmap(maze_matrix)
    p = scatter!(hm, poses, color=colors)
    savefig(p, joinpath(dirname, "maze.png"))
end



r2(x) = @sprintf "%6.6s" string(round(x, digits=2))
function plot_bcs(dirname::String, ::Dict, bcs::Vector)
    # compute mins, max, means, and stds for each dimension in 
    @assert length(bcs) > 0
    mins  = r2.([minimum([(bc[i]) for bc in bcs]) for i in 1:length(bcs[1])])
    maxs  = r2.([maximum([(bc[i]) for bc in bcs]) for i in 1:length(bcs[1])])
    means = r2.([mean([(bc[i]) for bc in bcs]) for i in 1:length(bcs[1])])
    stds  = r2.([std([(bc[i]) for bc in bcs]) for i in 1:length(bcs[1])])

    # TODO fix this if needed
    moves = [@sprintf "%6.6s" s for s in 
        ["UP", "DOWN", "LEFT", "RIGHT",
        "PICK_1", "PLACE_1", "PICK_2",
        "PLACE_2", "NOOP"]]
    headers = [@sprintf "%6.6s" s for s in ["dir", "min", "max", "mean", "std"]]
    open(joinpath(dirname, "stats.txt"), "w") do file
        println(file, join(headers, " "))
        for i in 1:length(bcs[1])
            println(file, "$(moves[i]) $(mins[i]) $(maxs[i]) $(means[i]) $(stds[i])")
        end
    end
end

