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
        fits::Vector{<:Real}, num_elites::Int, γ::AbstractFloat)
    """walks::{Vector{NTuple{2, Float64}}}"""
    # Make sure following 2 lines is the same as src/ga.jl::create_next_pop()
    num_elite_explorers = floor(Int, γ * num_elites)
    num_elite_exploiters = num_elites - num_elite_explorers
    elite_exploiter_idxs = sortperm(fits, rev=true)[1:num_elite_exploiters]
    elite_explorer_idxs = sortperm(novs, rev=true)[1:num_elite_explorers]
    colors = :blue
    @assert length(novs) == length(walks) == length(fits)
    color_fits = copy(fits) .- minimum(fits)
    max_fit = max(maximum(color_fits), 0.1)
    color_novs = copy(novs) .- minimum(novs)
    max_nov = max(maximum(color_novs), 0.1)
    function col(fit::AbstractFloat, nov::AbstractFloat)
        color = colorant"white"*0.2 + colorant"blue"*(nov/max_nov)*0.8 + colorant"white"*(fit/max_fit)*0.8
        # not really a better way to do this afaik
        r = clamp(red(color), 0, 1)
        g = clamp(green(color), 0, 1)
        b = clamp(blue(color), 0, 1)
        RGB(r, g, b)
    end
    colors = [col(fit, nov) for (nov, fit) in zip(color_novs, color_fits)]
    hm = heatmap(grid, colorbar = false, background_color=colorant"black", foreground_color=colorant"white")

    for i in sortperm(fits)
        marker_shape=:none
        marker_color=:match
        if i in elite_exploiter_idxs
            marker_shape=:hexagon
            marker_color=:white
        elseif i in elite_explorer_idxs
            println("painting elite explorer")
            marker_shape=:hexagon
            marker_color=:blue
        end
        offset_walk = [(1+p[1], 1+p[2]) for p in walks[i]]
        plot!(hm, offset_walk, legend = false, xticks=[], yticks=[], color=colors[i], markershape=marker_shape, markercolor=marker_color)
    end
    savefig(hm, name)
end

function plot_grid_and_walks(env,
        name::String,
        grid::AbstractArray,
        walks::Vector,
        novs::Vector{<:Real},
        fits::Vector{<:Real},
        num_elites::Int, γ::AbstractFloat)
    """walks::{Vector{NTuple{2, Float64}}}"""
    grid = prep_grid(env, grid)
    plot_walks(name, grid, walks, novs, fits, num_elites, γ)
end

function vis_outs(dirname::String, islocal::Bool)
    islocal && return
    run(`sbatch run-batch-vis.sh $dirname`, wait=false)
end
