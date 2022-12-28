function prep_grid(::MazeEnv, grid::AbstractArray{<:Real})
    grid
end

function prep_grid(::Dict, grid::AbstractArray{<:Real})
    food_1 = grid[:,:,1,end]
    food_2 = grid[:,:,2,end] * 4
    food_mat = food_1 .+ food_2
    food_mat *= 100
    food_mat
end

function plot_walks(name::String,
        grid::AbstractArray{<:Real},
        walks::Vector{Vector{NTuple{2, Float64}}},
        novs::Vector{<:Real},
        fits::Vector{<:Real})
    colors = :blue
    @assert length(novs) == length(walks) == length(fits)
    max_nov = max(maximum(novs), 0.1)
    color_fits = copy(fits) .- minimum(fits)
    max_fit = max(maximum(color_fits), 0.1)
    widths = [3*nov/max_nov for nov in novs]
    colors = [colorant"blue"*0.8 + colorant"yellow" * (fit/max_fit) for fit in color_fits]
    hm = heatmap(grid, colorbar = false, background_color=colorant"black", foreground_color=colorant"white")

    for (i, walk) in enumerate(walks)
        offset_walk = [p for p in walk]
        plot!(hm, offset_walk, legend = false, xticks=[], yticks=[], color=colors[i], linewidth=widths[i])
    end
    savefig(hm, name)

end

function plot_grid_and_walks(env,
        name::String,
        grid::AbstractArray,
        walks::Vector{Vector{NTuple{2, Float64}}},
        novs::Vector{<:Real},
        fits::Vector{<:Real})
    grid = prep_grid(env, grid)
    plot_walks(name, grid, walks, novs, fits)
end

function vis_outs(dirname::String, islocal::Bool)
    islocal && return
    run(`sbatch run-batch-vis.sh $dirname`, wait=false)
end
