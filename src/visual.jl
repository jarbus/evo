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

plot_walks(::String, ::Nothing, ::Any) = nothing
function plot_walks(name::String, initial_table::AbstractArray{Float32}, walks::Vector{Vector{NTuple{2, Float64}}})
    food_1 = initial_table[:,:,1,end]
    food_2 = initial_table[:,:,2,end] * 4
    food_mat = food_1 .+ food_2
    food_mat *= 100

    hm = heatmap(food_mat, colorbar = false, background_color=colorant"black", foreground_color=colorant"white")
    xlims!(0.5, size(food_mat, 1))
    ylims!(0.5, size(food_mat, 2))
    for walk in walks
        offset_walk = [1.5 .+ p for p in walk]
        plot!(hm, offset_walk, legend = false, xticks=[], yticks=[])
    end
    savefig(hm, name)
end



r2(x) = @sprintf "%6.6s" string(round(x, digits=2))
function plot_bcs(dirname::String, ::Dict, bcs::Vector, ::Vector=[])
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

function vis_outs(dirname::String, islocal::Bool)
    islocal && return
    run(`sbatch run-batch-vis.sh $dirname`, wait=false)
end
