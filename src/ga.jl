module GANS
using StableRNGs
using Flux
using Random
using Statistics
using Distributed
using Infiltrator
using NearestNeighbors
export compute_novelty, compute_novelties,
bc1, bc2, bc3, create_next_pop, add_to_archive!,
reorder!, dist, M, elite, mr, create_rollout_groups, 
compute_prefixes, decompress_group, add_elite_idxs,
compress_pop, all_v_all, singleton_groups

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

elite(x) = length(x) > 2 ? x[1:end-2] : x
# mr(x) = length(x) > 1 ? x[end-1] : 10.0 ^ rand(-2:-1:-5)
# M(x) = clamp(x*(2^(rand()*2-1)), 10.0^-5, 10.0^-2)
mr(x) = 0.001f0
M(x) = 0.001f0
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
    Fσs = [0.001f0]
    Nσs = [0.001f0]
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

function find_last_matching_idx(v1::Vector, v2::Vector)
    for i in 1:2:min(length(v1), length(v2))
        if v1[i] != v2[i]
            i > 1 && return i-2
            i == 1 && return 0
        end
    end
    return min(length(v1), length(v2))
end

function compute_elite_idxs(elites)
    """
    For each elite seed, compute all indexes i where elite_seed[1:i] is another elite. This can be
    used for caching results of reconstructions for other elites
    """
    eseeds = [e[:seeds] for e in elites]
    elite_idxs = Dict()
    for e1 in eseeds
        idxs = Set{Int}()
        for e2 in eseeds
            push!(idxs, find_last_matching_idx(e1, e2))
        end
        elite_idxs[e1] = idxs
    end
    elite_idxs
end

function add_elite_idxs(pop, elites)
    """
    groups: Vector of Tuples t, t[i][1] is identifier , t[i][2] is seed
    elites: vector of dicts
    returns: 
    groups_with_idxs: Vector of Tuples t, t[i][1] is identifier , t[i][2] is seed, t[i][3] is a set of idxs
    where 1:idx is an elite, whose reconstruction should be cached
    """
    elite_idxs = compute_elite_idxs(elites)
    new_pop = []
    for seed in pop
        if haskey(elite_idxs, seed)
            push!(new_pop, (seed, elite_idxs[seed]))
        elseif haskey(elite_idxs, elite(seed))
            push!(new_pop, (seed, elite_idxs[elite(seed)]))
        else
            push!(new_pop, (seed, Set{Int}()))
        end
    end
    new_pop
end

function compute_prefixes(elites; k::Int=10)
    """
    Returns the prefixes of elite seeds with the greatest 
    length(prefix)*number_of_elites_with_prefix
    """
    eseeds = [e[:seeds] for e in elites]
    # dict of prefixes => chars_reduced
    chars_reduced = Dict()
    for e1 in eseeds, e2 in eseeds
        e1 == e2 && continue
        min_len = min(length(e1), length(e2))
        # go up e1 and e2 until they stop matching
        for i in 1:2:min_len
            # prefix is the string up until they diverge, or one of them ends
            if e1[i] != e2[i] || i == min_len
                i == 1 && continue # skip if they diverge at the first char
                idx = e1[i] != e2[i] ? i-2 : min_len
                prefix = e1[1:idx]
                # skip if we've already seen this prefix on a different e1,e2 pair
                haskey(chars_reduced, prefix) && break 
                # once we find a unique prefix, go over all elites to 
                # compute how many chars it might save
                for e3 in eseeds
                    length(e3) < length(prefix) && continue
                    if e3[1:length(prefix)] == prefix
                        chars_reduced[prefix] = get(chars_reduced, prefix, 0) + length(prefix)
                    end
                end
                break
            end
        end
    end
    # sort by number of characters reduced, filter out top k, 
    chars_reduced_and_prefix = sort([(chrs, prefix) for (prefix, chrs) in chars_reduced], rev=true)
    chars_reduced_and_prefix = chars_reduced_and_prefix[1:min(k, length(chars_reduced_and_prefix))]
    chars_reduced = [char_and_pre[1] for char_and_pre in chars_reduced_and_prefix]
    @info "Characters reduced with new prefix: $chars_reduced"
    best_prefixes = [char_and_pre[2] for char_and_pre in chars_reduced_and_prefix]
    Dict(string(hash(pre))=>pre for pre in best_prefixes)
end


function compress_pop(pop, elites, prefixes)
    """
    groups: Vector of Tuples t, t[i][1] is identifier , t[i][2] is seed, t[i][3] is a set of idxs
    returns: new_groups, prefix_dict with string(hash(seed)) as key
    """
    new_pop = add_elite_idxs(pop, elites)
    # we check prefixes in order of decreasing length in order to maximize 
    # the number of characters we can replace
    prefixes_by_len = copy(sort([(length(v), k, v) for (k, v) in prefixes], rev=true))
    compressed_pop = []
    for (seed, idx) in new_pop
        for (len, id, prefix) in prefixes_by_len
            if length(seed) >= len && seed[1:len] == prefix
                seed = vcat(id, seed[len+1:end])
                break
            end
        end
        push!(compressed_pop, (seed, idx))
    end
    compressed_pop
end

function decompress_group(group, prefixes)

    new_group = []
    for (id, seeds, elite_idxs) in group
        if typeof(seeds[1]) == String
            prefix_seed = prefixes[seeds[1]]
            seeds = Float32.(vcat(prefix_seed, seeds[2:end]))
        end
        push!(new_group, (id, seeds, elite_idxs))
    end
    new_group
end
end

