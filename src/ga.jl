using StableRNGs
using Flux
using Random
using Statistics
using Distributed
using Infiltrator
using NearestNeighbors
dist(a, b) = sum((a .- b).^2)

function add_to_archive!(pops::Vector{Pop}, prob)
  for pop in pops
    @inline add_to_archive!(pop, prob)
  end
end
function add_to_archive!(pop::Pop, prob)
  for i in 1:length(pop.inds)
    if rand() <= prob
      push!(pop.archive, pop.inds[i].bc)
    end
  end
end

# mr(x) = length(x) > 1 ? x[end-1] : 10.0 ^ rand(-2:-1:-5)
# M(x) = clamp(x*(2^(rand()*2-1)), 10.0^-5, 10.0^-2)
mr(x) = 0.002f0
M(x) = 0.002f0

function bcs(pop::Pop)
    _bcs = Vector{BC}(undef, pop.size)
    @simd for i in 1:pop.size
        @inbounds _bcs[i] = pop.inds[i].bc
    end
    _bcs
end
function fitnesses(pop::Pop)
    fitnesses = Vector{F}(undef, pop.size)
    @simd for i in 1:pop.size
        @inbounds fitnesses[i] = pop.inds[i].fitness
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

novelties(pop::Pop) = novelties(pop.inds)
function novelties(inds::Vector{Ind})
    novelties = Vector{F}(undef, length(inds))
    @simd for i in 1:length(inds)
        @inbounds novelties[i] = inds[i].novelty
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
    # NOTE: WE USE THE HARD-CODED MUT IN NT INSTEAD
    Fσs = [0.002f0] # fitness mutation rate
    Nσs = [0.002f0] # novelty mutation rate
    exploiter_elites = make_elites(fitnesses(pop), nums[:n_e_exploit])
    explorer_elites  = make_elites(novelties(pop), nums[:n_e_explore])

    # TODO CHANGE THIS BACK
    # WE ARE RUNNING EXPERIMENTS ASSUMING γ ∈ [0, 1]
    # THIS BREAKS THE VISUALIZATIONS
    @assert length(explorer_elites) == 0 || length(exploiter_elites) == 0
    elites = [exploiter_elites; explorer_elites]
    next_inds = [deepcopy(elites[1])]
    for _ in 1:(nums[:n_n_exploit]+nums[:n_e_exploit]) # add exploiters
        push!(next_inds, mutate(rand(exploiter_elites), Fσs[1]))
    end
    for _ in 1:(nums[:n_n_explore]+nums[:n_e_explore]) # add explorers
        push!(next_inds, mutate(rand(explorer_elites), Nσs[1]))
    end
    next_inds = next_inds[1:pop.size]
    @assert length(next_inds) == pop.size
    # set ids of next gen
    for (i, ind) in enumerate(next_inds)
        ind.id = pop.id*"_"*string(i)
    end
    next_pop = Pop(pop.id, pop.size, next_inds)
    next_pop.elites = elites
    next_pop.archive = pop.archive
    next_pop
end

function popn!(x, n::Int)
    [pop!(x) for _ in 1:n]
end

function all_v_all(pop1::Vector{RolloutInd}, pop2::Vector{RolloutInd}; kwargs...)
    [[ind1, ind2] for ind1 in pop1 for ind2 in pop2]
end


function pop_group!(pop1::Vector{RolloutInd},
    pop2::Vector{RolloutInd},
    group_size::Int)
  @assert group_size % 2 == 0
  [pop!(p) for p in (pop1,pop2) for _ in 1:(group_size/2)]
end

function all_v_best(pop1::Vector{RolloutInd},
  pop2::Vector{RolloutInd};
  rollouts_per_ind::Int,
  rollout_group_size::Int)
  n_elites = rollouts_per_ind
  @assert n_elites <= length(pop1)
  @assert length(pop1) == length(pop2)
  @assert rollout_group_size == 2
  n_groups = length(pop1) * n_elites * 2
  groups = Vector{Vector{RolloutInd}}(undef, n_groups)
  group_idx::Int = 0
  # Do all v best where all is from pop1, then all_v_best 
  # where all is from pop2
  for (all_pop, best_pop) in [(pop1, pop2), (pop2, pop1)]
    for ind_all in all_pop
      # The first agents in each group are the elites
      for i in 1:n_elites
        group_idx += 1
        ind_best = best_pop[i]
        groups[group_idx] = [ind_all, ind_best]
      end
    end
  end
  @assert group_idx == n_groups
  groups
end

function random_groups(pop1::Vector{RolloutInd},
  pop2::Vector{RolloutInd};
  rollout_group_size::Int,
  rollouts_per_ind::Int)
  # random matchups between members of pop1 and members of pop2
  # where each individual is paired approx the same # of times
  @assert length(pop1) == length(pop2)
  pop_size = length(pop1)
  n_groups = ceil(Int, rollouts_per_ind*pop_size/rollout_group_size)
  n_agents = n_groups * rollout_group_size
  n_duplicates = ceil(Int, n_agents / pop_size)
  dupop1 = repeat(pop1, n_duplicates) |> shuffle
  dupop2 = repeat(pop2, n_duplicates) |> shuffle
  @assert length(dupop1) >= n_agents
  groups = [pop_group!(dupop1, dupop2, rollout_group_size) for _ in 1:n_groups*2]
  groups
end

# function singleton_groups(pop::Pop)
#     [[ind] for ind in pop]
# end
function singleton_groups(rollout_inds::Vector{RolloutInd};
    rollout_group_size::Int, rollouts_per_ind::Int)
    @assert rollout_group_size == 1
    @assert rollouts_per_ind == 1
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
  _, dists = knn(kdtree, bc_mat[:,1:n_pop_bcs], k)
  @assert length(dists) == n_pop_bcs
  @assert length(bc_ids) == length(pop.inds)
  for (i, idxs) in bc_ids
    max_nov::Float32 = -1f0
    most_nov_bc::BC = BC()
    for idx in idxs
      if sum(dists[idx]) > max_nov
        max_nov = sum(dists[idx])
        most_nov_bc = bc_mat[:,idx]
      end
    end
    pop.inds[i].novelty = max_nov
    pop.inds[i].bc = most_nov_bc
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
    ind.fitness = mean(ind.fitnesses)
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
  compute_novelties!(pops)
  compute_fitnesses!(pops)
  add_to_archive!(pops, arxiv_prob)
  average_walks!(pops)
  for pop in pops, ind in pop.inds
      @assert length(ind.bcs) > 0
  end
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
