module GANS
using StableRNGs
using Flux
export reconstruct, compute_novelty, bc1, create_next_pop, add_to_archive!, reorder!

function reconstruct(x::Vector{<:UInt32}, len, ϵ=0.01)
  @assert length(x) > 0
  @assert len > 0
  theta = Flux.glorot_normal(StableRNG(x[1]), len)
  for seed in x
    theta .+= ϵ .* Flux.glorot_normal(StableRNG(seed), len)
  end

  @assert theta .|> isnan |> any |> !
  theta
end


function compute_novelty(ind_bc::Vector{<:Float64}, archive_and_pop::Vector{Vector{Float64}})::Float64
    # Assumptions: Novelty against self is zero, ind_bc is in archive_and_pop
    sum(sum((ind_bc .- bc) .^ 2) for bc in archive_and_pop) / (length(archive_and_pop) - 1)
end


function compute_novelty(ind_bc::Tuple, archive_and_pop::Vector)::Float64 
    # Assumptions: Novelty against self is zero, ind_bc is in archive_and_pop
    sum(sum(sum((ind_bc .- bc) .^ 2) for bc in archive_and_pop)) / (length(archive_and_pop) - 1)
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

end
