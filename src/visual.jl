function prep_grid(::MazeEnv, grid::AbstractArray{<:Real})
    grid'
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
        walks::Vector,
        novs::Vector{<:Real},
        fits::Vector{<:Real})
    """walks::{Vector{NTuple{2, Float64}}}"""
    colors = :blue
    @assert length(novs) == length(walks) == length(fits)
    color_fits = copy(fits) .- minimum(fits)
    max_fit = max(maximum(color_fits), 0.1)
    color_novs = copy(novs) .- minimum(novs)
    max_nov = max(maximum(color_novs), 0.1)
    colors = [colorant"white"*0.2 + colorant"blue"*(nov/max_nov)*0.8 + colorant"white"*(fit/max_fit)*0.8 for (nov, fit) in zip(color_novs, color_fits)]
    hm = heatmap(grid, colorbar = false, background_color=colorant"black", foreground_color=colorant"white")

    for i in sortperm(fits)
        offset_walk = [(1+p[1], 1+p[2]) for p in walks[i]]
        plot!(hm, offset_walk, legend = false, xticks=[], yticks=[], color=colors[i])
    end
    savefig(hm, name)

end

function plot_grid_and_walks(env,
        name::String,
        grid::AbstractArray,
        walks::Vector,
        novs::Vector{<:Real},
        fits::Vector{<:Real})
    """walks::{Vector{NTuple{2, Float64}}}"""
    grid = prep_grid(env, grid)
    plot_walks(name, grid, walks, novs, fits)
end

function vis_outs(dirname::String, islocal::Bool)
    islocal && return
    run(`sbatch run-batch-vis.sh $dirname`, wait=false)
end
