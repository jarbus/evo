function prep_grid(::MazeEnv, grid::AbstractArray{<:Real})
    grid'
end

function prep_grid(::Dict, grid::AbstractArray{<:Real})
    food_1 = grid[:,:,1,end]
    food_2 = grid[:,:,2,end] * 4
    food_mat = food_1 .+ food_2
    food_mat *= 100
    food_mat'
end

function plot_walks(name::String,
        grid::AbstractArray{<:Real},
        walks::Vector,
        novs::Vector{<:Real},
        fits::Vector{<:Real}, num_elites::Int, γ::AbstractFloat)
    """walks::{Vector{NTuple{2, Float64}}}"""
    nums = compute_ratios(length(walks), γ, num_elites)
    elite_exploiter_idxs = sortperm(fits, rev=true)[1:nums[:n_e_exploit]]
    elite_explorer_idxs = sortperm(novs, rev=true)[1:nums[:n_e_explore]]
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
        end
        offset_walk = [(1+p[1], 1+p[2]) for p in walks[i]]
        plot!(hm, offset_walk, legend = false, xticks=[], yticks=[], color=colors[i], markershape=marker_shape, markercolor=marker_color)
    end
    # paint explorers elites on top of fit
    marker_shape=:hexagon
    marker_color=:blue
    for i in elite_explorer_idxs
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
function plot_grid_and_walks(env,
        outroot::String,
        grid::AbstractArray,
        pops::Vector{Pop},
        num_elites::Int, γ::AbstractFloat)
    grid = prep_grid(env, grid)
    for (i, pop) in enumerate(pops)
      plot_walks(outroot*"-$i.png", grid, walks(pop), novelties(pop), fitnesses(pop), num_elites, γ)
    end
end

elite_color = colorant"blue"
pop_color = colorant"lightblue"
archive_color = colorant"pink"

function get_colors_and_bc(pop::Pop, n_elites::Int)
  colors = [[elite_color  for _ in 1:n_elites];
            [pop_color for _ in 1:length(pop.inds)-n_elites];
            [archive_color   for _ in 1:length(pop.archive)]]
  _bcs = [bcs(pop); collect(pop.archive)]
  @assert length(_bcs) == length(colors)
  _bcs, RGB.(colors)
end

Coord = Tuple{Float32, Float32}
function plot_bcs(outroot::String, pops::Vector{Pop}, n_elites::Int)
  "User interface for plotting bcs"
  for (i, pop) in enumerate(pops)
    "Change this function to change the bc handler"
    plot_4bcs(outroot*"-$i.png",pop, n_elites)
  end
end
function plot_8bcs(outroot::String, pop::Pop, n_elites::Int)
  titles="pick_0 pick_1 place_0 place_1 xpos ypos light pos_fit" |>
            split .|> string
  mins = [0,  0, 0,  0, 0, 0,  0, 0]
  maxs = [10,10,10, 10, 1, 1, 30,30]
  @assert length(titles) == length(mins) == length(maxs)
  _bcs, colors = get_colors_and_bc(pop, n_elites)
  plot_bcs(outroot, _bcs, colors,
            titles=titles, mins=mins, maxs=maxs)
  @assert length(_bcs[1]) == 8
end
function plot_9bcs(outroot::String, pop::Pop, n_elites::Int)
  titles="pick_0 pick_1 place_0 place_1 xpos ypos light health acts" |>
            split .|> string
  mins = [0,  0, 0,  0, 0, 0, -50,  0, 0]
  maxs = [10,10,10, 10, 1, 1,   0, 30,10]
  @assert length(titles) == length(mins) == length(maxs)
  _bcs, colors = get_colors_and_bc(pop, n_elites)
  plot_bcs(outroot, _bcs, colors,
            titles=titles, mins=mins, maxs=maxs)
  @assert length(_bcs[1]) == 9
end

function plot_4bcs(outroot::String, pop::Pop, n_elites::Int)
  titles="xpos ypos light health" |>
            split .|> string
  mins = [0, 0, -50,  0]
  maxs = [1, 1,   0, 30]
  @assert length(titles) == length(mins) == length(maxs)
  _bcs, colors = get_colors_and_bc(pop, n_elites)
  plot_bcs(outroot, _bcs, colors,
            titles=titles, mins=mins, maxs=maxs)
  @assert length(_bcs[1]) == 4
end
function plot_bcs(name::String,
    bcs::Vector{BC},
    colors::Vector{<:RGB};
    titles::Vector{String},
    mins::Vector{<:Real},
    maxs::Vector{<:Real})
  "core bc plotter, indifferent to the type of bc"
  bc_len = length(bcs[1])
  p = plot(layout=(1,bc_len),
           size=(125*bc_len, 300),
           background_color=colorant"black",
           foreground_color=colorant"white",
           legend=false)
  for bc_idx in eachindex(titles)
    coords=Vector{Coord}(undef, length(bcs))
    for (i, bc) in enumerate(bcs)
      coords[i] = (rand(Float32), bc[bc_idx])
    end
    scatter!(p[bc_idx], title=titles[bc_idx],
             coords, color=colors)
    # plot elites on top for better visibility
    elites_idxs = colors .== colorant"blue"
    scatter!(p[bc_idx], title=titles[bc_idx],
             coords[elites_idxs], color=colors[elites_idxs])
    xticks!(p[bc_idx], 0:-1)
    ylims!(p[bc_idx], mins[bc_idx], maxs[bc_idx])
  end
  savefig(p, name)
end

function vis_outs(dirname::String, islocal::Bool)
    islocal && return
    run(`sbatch run-batch-vis.sh $dirname`, wait=false)
end
