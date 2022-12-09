module GANS
using StableRNGs
using Flux
using Statistics
using Distributed
using Infiltrator
export reconstruct, compute_novelty, bc1,
create_next_pop, add_to_archive!, reorder!, 
average_bc, compute_elite

function compute_novelty(ind_bc::Vector{<:Float64}, archive_and_pop::Vector{<:Any}; k::Int=25)::Float64
    # Assumptions: Novelty against self is zero, ind_bc is in archive_and_pop
    @assert k < length(archive_and_pop)
    @assert length(ind_bc) == length(archive_and_pop[1])
    # Assumptions: Novelty against self is zero, ind_bc is in archive_and_pop
    dists = [sum((ind_bc .- bc) .^ 2f0) for bc in archive_and_pop]# / (length(archive_and_pop) - 1)
    sum(sort(dists, rev=true)[1:k]) / k
end


function compute_novelty(ind_bc::Tuple, archive_and_pop::Vector; k::Int=25)::Float64 
    # for mazes
    @assert k < length(archive_and_pop)
    @assert length(ind_bc) == length(archive_and_pop[1])
    # Assumptions: Novelty against self is zero, ind_bc is in archive_and_pop
    dists = [sum((ind_bc .- bc) .^ 2f0) for bc in archive_and_pop]# / (length(archive_and_pop) - 1)
    sum(sort(dists, rev=true)[1:k]) / k

end

function reorder!(novelties, F, BC, pop)
   # remove maximum element from list

   order = sortperm(novelties, rev=true)
   # elite_idx = argmax(F)
   # move elite to the front
   # deleteat!(order, findfirst(==(elite_idx), order))
   # @assert elite_idx ∉ order
   # pushfirst!(order, elite_idx)
   @assert novelties[order[2]] >= novelties[order[3]]
   # reorder lists in-place
   F[:] = F[order]
   BC[:] = BC[order]
   pop[:] = pop[order]
   novelties[:] = novelties[order]
   # @assert argmax(F) == 1
   @assert length(pop) == length(BC) == length(F)
end

function add_to_archive!(archive, BC, pop)
  for i in 1:length(pop)
      if i > 1 && rand() <= 0.01
          push!(archive, (BC[i], pop[i]))
      end
  end
end

function create_next_pop(gen::Int, pop::Vector{Vector{UInt32}}, num_elites::Int)
    pop_size = length(pop)
    @assert pop_size > 0
    @assert num_elites != pop_size

    next_pop = [copy(pop[k]) for k in 1:num_elites] # copy elites
    for i in (num_elites+1):pop_size
        parent_idx = (rand(UInt) % num_elites) + 1 # select parent
        push!(next_pop, copy(pop[parent_idx])) # copy parent to next pop
        push!(next_pop[i], rand(UInt32)) # mutate parent into child
    end
    @assert length(next_pop) == pop_size
    next_pop 
end

function bc1(x::Vector{<:Integer}, num_actions=9)::Vector{Float64}
    # count number of each element in x
    counts = zeros(Int, num_actions)
    for i in x
        counts[i] += 1
    end
    counts ./ length(x)
end

function average_bc(bcs::Vector)
  @assert Set(length.(bcs)) |> length == 1
  [mean(x) for x in zip(bcs...)]
end

function compute_elite(f, pop, F; k::Int=10, n::Int=30)
  # run n evals in parallel on top k members to compute elite
  top_F_idxs = sortperm(F, rev=true)[1:min(k, length(pop))]
  @assert F[top_F_idxs[1]] >= maximum(F)
  rollout_Fs = pmap(1:k*n) do rollout_idx
      # get member ∈ [1,10] from rollout count
      p = floor(Int, (rollout_idx-1) / n) + 1
      @assert p in 1:k
      fit = f(pop[top_F_idxs[p]], pop[top_F_idxs[p]])
      (fit[1] + fit[2])/2
  end
  @assert rollout_Fs isa Vector{<:AbstractFloat}
  accurate_Fs = [sum(rollout_Fs[i:i+n-1])/n for i in 1:n:length(rollout_Fs)]
  @assert length(accurate_Fs) == k
  elite_idx = argmax(accurate_Fs)
  elite = maximum(accurate_Fs), pop[top_F_idxs[elite_idx]]
  elite

end
end
