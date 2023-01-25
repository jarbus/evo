using StableRNGs
using Flux
using Random
using Statistics
using Distributed
using Infiltrator
using NearestNeighbors
dist(a, b) = sum((a .- b).^2)

function compute_novelty(ind_bc::Vector, archive_and_pop::Matrix; k::Int=25)::Float64 
    # for mazes
    @assert k < size(archive_and_pop, 2)
    @assert size(ind_bc,1) == size(archive_and_pop, 1)
    # Assumptions: Novelty against self is zero, ind_bc is in archive_and_pop
    kdtree = KDTree(archive_and_pop, leafsize=10)
    inds, dists = knn(kdtree, ind_bc, k+1)
    # @assert length(dists) == length(archive_and_pop) - k
    return sum(dists) / k
end


function compute_novelties(ind_bc::Matrix, archive_and_pop::Matrix; k::Int=25) 
    # for mazes
    @assert k < size(archive_and_pop, 2)
    @assert size(ind_bc, 2) <= size(archive_and_pop, 2)
    @assert size(ind_bc, 1) == size(archive_and_pop, 1)
    @assert size(ind_bc, 1) == size(archive_and_pop, 1)
    # Assumptions: Novelty against self is zero, ind_bc is in archive_and_pop
    kdtree = KDTree(archive_and_pop, leafsize=10)
    inds, dists = knn(kdtree, ind_bc, k+1)
    # @assert length(dists) == length(archive_and_pop) - k
    return [sum(d) / k for d in dists]
end


function reorder!(to_sort, vecs...)
   """sorts all vecs according to the order
   specified by to_sort"""


   order = sortperm(to_sort, rev=true)
   length(order) > 2 && @assert to_sort[order[2]] >= to_sort[order[3]]
   # reorder lists in-place
   for vec in vecs
       @assert length(vec) == length(order)
       vec[:] = vec[order]
   end
   to_sort[:] = to_sort[order]
end

function add_to_archive!(archive, BC, pop, prob)
  for i in 1:length(pop)
      if i > 1 && rand() <= prob
          push!(archive, (BC[i], pop[i]))
      end
  end
end

# mr(x) = length(x) > 1 ? x[end-1] : 10.0 ^ rand(-2:-1:-5)
# M(x) = clamp(x*(2^(rand()*2-1)), 10.0^-5, 10.0^-2)
mr(x) = 0.002f0
M(x) = 0.002f0
function create_next_pop(gen::Int,
        sc,
        pop::Vector{Vector{Float32}},
        fitnesses::Vector{<:AbstractFloat},
        novelties::Vector{<:AbstractFloat},
        bcs::Vector{<:Vector{<:AbstractFloat}},
        γ::Float64,
        num_elites::Int)
    pop_size = length(pop)
    @assert length(pop) == length(fitnesses) == length(novelties) == length(bcs)
    @assert 0 < num_elites < pop_size
    @assert pop_size > 0
    new_pop_size = pop_size - num_elites
    # If you change the below two lines, update src/visual.jl::plot_walks() too
    num_elite_explorers = floor(Int, γ * num_elites)
    num_elite_exploiters = num_elites - num_elite_explorers
    @assert num_elite_explorers + num_elite_exploiters == num_elites
    num_next_explorers = floor(Int, new_pop_size*(num_elite_explorers / num_elites))
    num_next_exploiters  = new_pop_size - num_next_explorers
    @assert num_elite_explorers + num_elite_exploiters + new_pop_size == pop_size
    function make_elites(order_metric, num)
        order = sortperm(order_metric, rev=true)
        elites = [Dict(
          :seeds => pop[order[i]],
          :bc => bcs[order[i]],
          :novelty => novelties[order[i]],
          :fitness => fitnesses[order[i]],
         ) for i in 1:num]
    end
    # if gen == 1
        # Fσs = [0.002f0 for _ in 1:num_elite_exploiters]
        # Nσs = [0.002f0 for _ in 1:num_elite_explorers]
    Fσs = [0.002f0]
    Nσs = [0.002f0]
    # else
    #     σs = [mr(pop[i]) for i in (1+num_elites):pop_size]
    #     ΔFs = [f - sc[elite(pop[i])][:fitness] for (i,f) in enumerate(fitnesses[1+num_elites:end])]
    #     ΔNs = [dist(bc, sc[elite(pop[i])][:bc]) for (i,bc) in enumerate(bcs[1+num_elites:end])]
    #     Fσs = σs[sortperm(ΔFs, rev=true)][1:num_elite_exploiters]
    #     Nσs = σs[sortperm(ΔNs, rev=true)][1:num_elite_explorers]
    # end
    exploiter_elites = make_elites(fitnesses, num_elite_exploiters)
    explorer_elites  = make_elites(novelties, num_elite_explorers)

    next_pop::Vector{Vector{Float32}} = vcat(
                    [copy(e[:seeds]) for e in exploiter_elites],
                    [copy(e[:seeds]) for e in explorer_elites]
                    )
    @assert length(next_pop) == num_elites
    num_next_exploiters > 0 && for _ in 1:num_next_exploiters
        push!(next_pop, copy(rand(exploiter_elites)[:seeds]))
        push!(next_pop[end], M(rand(Fσs)), rand(UInt32))
    end
    num_next_explorers > 0 && for _ in 1:num_next_explorers
        push!(next_pop, copy(rand(explorer_elites)[:seeds]))
        push!(next_pop[end], M(rand(Nσs)), rand(UInt32))
    end
    elites::Vector{Dict} = vcat(exploiter_elites, explorer_elites)
    @assert length(next_pop) == pop_size
    @assert length(elites) == num_elites
    @assert exploiter_elites[1][:seeds] == next_pop[1]
    @assert exploiter_elites[num_elite_exploiters][:seeds] == next_pop[num_elite_exploiters]
    if length(explorer_elites) > 1
        @assert explorer_elites[1][:seeds] == next_pop[1+num_elite_exploiters]
        @assert explorer_elites[num_elite_explorers][:seeds] == next_pop[num_elite_explorers+num_elite_exploiters]
    end

    next_pop, elites
end

function bc1(x::Vector{<:Integer}, num_actions=9)::Vector{Float64}
    # percentage of each action
    counts = zeros(Int, num_actions)
    for i in x
        counts[i] += 1
    end
    counts ./ length(x)
end


function bc2(x::Vector{<:Vector{<:Integer}}, num_actions=9)::Vector{Float64}
    # percentage of each action for num_actions windows
    bcs = []
    i = 1
    while i <= length(x)
        i % num_actions == 1 && push!(bcs, zeros(Float64, num_actions))
        for j in x[i]
            bcs[end][j] += 1
        end
        if i % num_actions == 0 
            bcs[end] /= sum(bcs[end])
        end

        i += 1
    end
    bcs[end] /= sum(bcs[end])
    bc = vcat(bcs...)
    # @assert length(bc) == floor(Int, length(x) / num_actions) 
    bc 
end

function bc3(avg_walks::Vector{NTuple{2, Float64}}, fitness::Float32)::Vector{Float64}
    # starting position, average position, ending position and fitness
    start_pos = avg_walks[1]
    mean_x = mean([x for (x,y) in avg_walks])
    mean_y = mean([y for (x,y) in avg_walks])
    mean_pos = (mean_x, mean_y)
    end_pos = avg_walks[end]
    [start_pos..., mean_pos..., end_pos..., fitness]
end


function popn!(x, n::Int)
    [pop!(x) for _ in 1:n]
end



function create_rollout_groups(pop::Vector,
        eseeds::Vector,
        rollout_group_size::Int, rollouts_per_ind::Int)
    """Creates a vector of groups to evaluate. Each group is half elites from
    the previous generation and half members of the current population. 
    Each pop member and elite are sampled an equal amount, plus or minus 1.

    # Return a vector of Dict{Int, Vector}s where 
        Integer key is the positive index in the population
    or negative index in the elite vector
        Vector `v` is length 2, where `v[1]` is the count of how many times
    this member has been chosen and `v[2]` is the member seed itself.
    """
    if rollout_group_size == 1 || eseeds == []
        return create_rollout_groups(pop, rollout_group_size, rollouts_per_ind)
    end

    function make_group!(g_pop, g_elites, n)
        n_pop = ceil(Int, n/2)
        n_elite = n - n_pop
        @assert n_pop <= length(g_pop)
        @assert n_elite <= length(g_elites)
        # ints from pop are positive, ints from elites are negative
        group_pop_ids = popn!(g_pop, n_pop)
        group_elite_ids = popn!(g_elites, n_elite)
        group_pop = [(i, pop[i]...) for i in group_pop_ids]
        group_elite = [(-i, eseeds[i]...) for i in group_elite_ids]
        group = vcat(group_pop, group_elite) |> shuffle
        group
    end

    n_groups = ceil(Int, rollouts_per_ind*2*length(pop)/rollout_group_size)
    n_agents = n_groups * rollout_group_size
    if n_agents < length(pop)
        error("Creating $n_groups groups of size $rollout_group_size will not include all members of the population.")
    end
    n_pop_agents = n_elite_agents = ceil(Int, n_agents / 2)
    n_pop_duplicates = ceil(Int, n_pop_agents / length(pop))
    n_elite_duplicates = ceil(Int, n_elite_agents / length(eseeds))
    massive_pop = vcat([collect(1:length(pop)) for _ in 1:n_pop_duplicates]...) |> shuffle
    massive_elites = vcat([collect(1:length(eseeds)) for _ in 1:n_elite_duplicates]...) |> shuffle
    @assert length(massive_pop) >= n_pop_agents
    @assert length(massive_elites) >= n_elite_agents
    groups = [make_group!(massive_pop, massive_elites, rollout_group_size) for _ in 1:n_groups]
    #counts = Dict()
    #for g in groups
    #    for (i, _) in g
    #        i < 0 && continue
    #        counts[i] = get(counts, i, 0) + 1
    #    end
    #end
    #for count in values(counts)
    #    @assert count == rollouts_per_ind || count == rollouts_per_ind + 1
    #end
    groups
end

function one_v_self(pop::Vector)
    [[(i, p1...), (i, p1...)] for (i, p1) in enumerate(pop)]
end

function all_v_all(pop::Vector)
    [[(i, p1...), (j, p2...)] for (i, p1) in enumerate(pop) for (j, p2) in enumerate(pop)]
end

function singleton_groups(pop::Vector)
    [[(i, p...)] for (i, p) in enumerate(pop)]
end

function create_rollout_groups(pop::Vector,
        rollout_group_size::Int, rollouts_per_ind::Int)

    n_groups = ceil(Int, rollouts_per_ind*length(pop)/rollout_group_size)
    function make_group!(g_pop, n)
        group_pop_ids = popn!(g_pop, n)
        group_pop = [(i, pop[i]...) for i in group_pop_ids] |> shuffle
        group_pop
    end
    n_agents = n_groups * rollout_group_size
    n_duplicates = ceil(Int, n_agents / length(pop))
    massive_pop = vcat([collect(1:length(pop)) for _ in 1:n_duplicates]...) |> shuffle
    @assert length(massive_pop) >= n_agents
    groups = [make_group!(massive_pop, rollout_group_size) for i in 1:n_groups]
    # counts = Dict()
    # for g in groups
    #     for (i, _) in g
    #         i < 0 && continue
    #         counts[i] = get(counts, i, 0) + 1
    #     end
    # end
    # for count in values(counts)
    #     @assert count == rollouts_per_ind || count == rollouts_per_ind + 1
    # end
    groups
end
