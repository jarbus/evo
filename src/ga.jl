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


function add_to_archive!(archive, BC, pop, prob)
  for i in 1:length(pop)
      if i > 1 && rand() <= prob
          push!(archive, (BC[i], pop[i]))
      end
  end
end
function add_to_archive!(pops::Vector{Pop}, prob)
  for pop in pops
    @inline add_to_archive!(pop, prob)
  end
end
function add_to_archive!(pop::Pop, prob)
  for i in 1:length(pop.inds), bc in pop.inds[i].bcs
    if rand() <= prob
      push!(pop.archive, bc)
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

function fitnesses(pop::Pop)
    fitnesses = Vector{F}(undef, pop.size)
    @simd for i in 1:pop.size
        @assert length(pop.inds[i].fitnesses) == 1
        @inbounds fitnesses[i] = pop.inds[i].fitnesses[1]
    end
    fitnesses
end

function walks(pop::Pop)
    walks = Vector{Vector}(undef, pop.size)
    @simd for i in 1:pop.size
        @inbounds walks[i] = pop.inds[i].walks[1]
    end
    walks
end

function novelties(pop::Pop)
    novelties = Vector{F}(undef, pop.size)
    @simd for i in 1:pop.size
        @inbounds novelties[i] = pop.inds[i].novelty
    end
    novelties
end

function compute_ratios(pop_size::Int, γ::Float64, num_elites::Int)
    """Compute % of elites and pop to select based on fitness
    and novelty for a given exploration ratio γ."""
    # If you change the below two lines, update src/visual.jl::plot_walks() too
    new_pop_size = pop_size - num_elites
    num_elite_explorers = floor(Int, γ * num_elites)
    num_elite_exploiters = num_elites - num_elite_explorers
    @assert num_elite_explorers + num_elite_exploiters == num_elites
    num_next_explorers = floor(Int, new_pop_size*(num_elite_explorers / num_elites))
    num_next_exploiters  = new_pop_size - num_next_explorers
    @assert num_elite_explorers + num_elite_exploiters + new_pop_size == pop_size
    Dict(:n_e_explore=>num_elite_explorers,
         :n_e_exploit=>num_elite_exploiters,
         :n_n_explore=>num_next_explorers,
         :n_n_exploit=>num_next_exploiters)
end

function mutate(ind::Ind, mut_rate::Float32)
    """Mutate an individual's genome."""
    @assert length(ind.geno) > 0
    child = Ind(ind)
    push!(child.geno, mut_rate, rand(UInt32))
    child
end

function create_next_pop(pops::Vector{Pop}, γ::Float64, num_elites::Int)
    @inline [create_next_pop(pop, γ, num_elites) for pop in pops]
end
function create_next_pop(pop::Pop, γ::Float64, num_elites::Int)
    nums = compute_ratios(pop.size, γ, num_elites)
    function make_elites(order_metric, num)
        Ind.(pop.inds[sortperm(order_metric, rev=true)[1:num]])
    end
    Fσs = [0.002f0] # fitness mutation rate
    Nσs = [0.002f0] # novelty mutation rate
    exploiter_elites = make_elites(fitnesses(pop), nums[:n_e_exploit])
    explorer_elites  = make_elites(novelties(pop), nums[:n_e_explore])

    elites = [exploiter_elites; explorer_elites]
    next_inds = deepcopy(elites)
    for _ in 1:nums[:n_n_exploit] # add exploiters
        push!(next_inds, mutate(rand(exploiter_elites), Fσs[1]))
    end
    for _ in 1:nums[:n_n_explore] # add explorers
        push!(next_inds, mutate(rand(explorer_elites), Nσs[1]))
    end
    # set ids of next gen
    for (i, ind) in enumerate(next_inds)
        ind.id = pop.id*"_"*string(i)
    end
    next_pop = Pop(pop.id, pop.size, next_inds)
    next_pop.elites = elites
    next_pop
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
function all_v_all(pop1::Vector{RolloutInd}, pop2::Vector{RolloutInd})
    [[ind1, ind2] for ind1 in pop1 for ind2 in pop2]
end

# function singleton_groups(pop::Vector)
#     [[(i, p...)] for (i, p) in enumerate(pop)]
# end
# function singleton_groups(pop::Pop)
#     [[ind] for ind in pop]
# end
function singleton_groups(rollout_inds::Vector{RolloutInd})
  [[ro] for ro in rollout_inds]
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

function make_bc_matrix(pop::Pop)
  """Compute the number of BCs we want to compare with knn
  Returns:
    bc_matrix: a matrix of BCs, where each row is a BC
    bc_ids: a dictionary mapping each ind to the row idx of its BCs
    n_pop_bcs: the first 1:n_pop_bcs rows of the bc_matrix are
    the elements we want to compute knn for
  """
  n_pop_bcs::Int = 0
  bc_len = length(pop.inds[1].bcs[1])
  @simd for ind in pop.inds
    n_pop_bcs += length(ind.bcs)
  end
  all_bcs = n_pop_bcs + length(pop.archive)
  bc_ids = Dict{Int, Vector{Int}}()
  bc_matrix = Array{Float32, 2}(undef, bc_len, all_bcs)
  idx::Int = 0
  for i in eachindex(pop.inds), bc in pop.inds[i].bcs
    idx += 1
    bc_matrix[:, idx] = bc
    if haskey(bc_ids, i)
      push!(bc_ids[i], idx)
    else
      bc_ids[i] = [idx]
    end
  end
  for bc in pop.archive
    idx += 1
    bc_matrix[:, idx] = bc
  end
  @assert idx == all_bcs
  bc_matrix, bc_ids, n_pop_bcs
end

compute_novelties!(pops::Vector{Pop}) = map(compute_novelties!, pops)
function compute_novelties!(pop::Pop, k=25)
  """Compute maximum novelty over all bcs for each ind in pop"""
  bc_mat, bc_ids, n_pop_bcs = EvoTrade.make_bc_matrix(pop)
  k = min(size(bc_mat,2), k)
  idx::Int = 0
  
  kdtree = KDTree(bc_mat, leafsize=10)
  inds, dists = knn(kdtree, bc_mat[:,1:n_pop_bcs], k)
  @assert length(dists) == n_pop_bcs
  @assert length(bc_ids) == length(pop.inds)
  for (i, idxs) in bc_ids
    max_nov::Float32 = 0f0
    most_nov_bc::BC = BC()
    for idx in idxs
      if sum(dists[idx]) > max_nov
        max_nov = sum(dists[idx])
        most_nov_bc = bc_mat[:,idx]
      end
    end
    pop.inds[i].novelty = max_nov
    pop.inds[i].bcs = [most_nov_bc]
  end
end

average_walks!(pops::Vector{Pop}) = map(average_walks!, pops)
function average_walks!(pop::Pop)
  """Compute the average walk for each ind in pop"""
  for ind in pop.inds
    avg_walk::Vector{Tuple{Float32, Float32}} = []
    for step in zip(ind.walks...)
        avg_step = mean.(zip(step...)) .|> Float32
        push!(avg_walk, tuple(avg_step...))
    end
    ind.walks = [avg_walk]
  end
end
compute_fitnesses!(pops::Vector{Pop}) = map(compute_fitnesses!, pops)
function compute_fitnesses!(pop::Pop)
  """This replaces all individual fitness scores
  with a singe fitness score."""
  for ind in pop.inds
    ind.fitnesses = Vector{Float32}([mean(ind.fitnesses)])
  end
end

function update_pop!(pop::Pop, batch::Batch)
  """Update the info of each ind with info from batch"""
  for k in keys(batch.rews)
    !haskey(pop.id_map, k) && continue
    idx = pop.id_map[k]
    push!(pop.inds[idx].fitnesses, batch.rews[k]...)
    push!(pop.inds[idx].bcs, batch.bcs[k]...)
    push!(pop.inds[idx].walks, batch.info["avg_walks"][k]...)
  end
end

function update_pops!(pops::Vector{Pop}, batches::Vector{Batch}, arxiv_prob::Float32=0.01f0)
  """Update the info of each ind with info from batch"""
  for batch in batches
    for pop in pops
      @inline update_pop!(pop, batch)
    end
  end
  for pop in pops, ind in pop.inds
      @assert length(ind.bcs) > 0
  end
  add_to_archive!(pops, arxiv_prob)
  compute_novelties!(pops)
  compute_fitnesses!(pops)
  average_walks!(pops)
end

function update_pops!(pop::Pop, batches::Vector{Batch})
  """Update the info of each ind with info from batch"""
  for batch in batches
    @inline update_pop!(pop, batch)
  end
end

function most_novel(pop::Pop)
  @inline most_nov_idx = argmax(novelties(pop))
  pop.inds[most_nov_idx]
end

function most_fit(pop::Pop)
  @inline most_fit_idx = argmax(fitnesses(pop))
  pop.inds[most_fit_idx]
end
