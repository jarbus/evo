module NoiseTables
export NoiseTable, compute_grad, get_noise, reconstruct, SeedCache, cache_elites!
using StableRNGs
using Flux
using LRUCache
struct NoiseTable
  rng::StableRNGs.LehmerRNG
  noise::Vector{Float32}
  nparams::Int
  pop_size::Int
  σ::Float32
end

NoiseTable(rng::StableRNGs.LehmerRNG, nparams::Int, pop_size::Int, σ::Float32) = NoiseTable(rng, σ .* Flux.glorot_normal(rng, nparams + pop_size), nparams, pop_size, σ)
get_noise(nt::NoiseTable, idx::Int) = get_noise(nt, UInt32(idx))
function get_noise(nt::NoiseTable, idx::UInt32)
  idx = idx % UInt32(nt.pop_size) + 1
  @view nt.noise[idx:idx+nt.nparams-1]
end

function compute_grad(nt::NoiseTable, centered_ranks::Vector{Float32})
  @assert nt.pop_size == length(centered_ranks)
  grad = zeros(Float32, nt.nparams)
  for i in 1:nt.pop_size
    grad += get_noise(nt, i) * centered_ranks[i]
  end
  grad
end

function reconstruct(nt::NoiseTable, seeds::Vector{<:UInt32}, ϵ::Float32=0.01f0)
  theta = zeros(Float32, nt.nparams)
  theta .+= @inline @views @inbounds get_noise(nt, seeds[1])
  for seed in seeds[2:end]
    @inline @views @inbounds theta .+= get_noise(nt, seed)
  end
  theta *= ϵ
  theta
end

SeedCache = LRU{Vector{UInt32},Vector{Float32}}

function cache_elites!(param_cache::SeedCache, nt::NoiseTable, elites::Vector{Vector{UInt32}}, ϵ::Float32=0.01)
  """On each processor, cache the parameters for all elites used to create the next 
  generation this makes reconstruction per generation O(T), where T is the truncation size
  """
  for elite in elites
    @inbounds param_cache[elite] = reconstruct(param_cache, nt, elite, ϵ)
  end
end

function reconstruct(param_cache::SeedCache, nt::NoiseTable, seeds::Vector{UInt32}, ϵ::Float32=0.01f0)
  """Reconstruction function that finds the nearest cached ancestor and
  reconstructs all future generations. Creates a new parameter vector if
  no ancestor is found.
  """
  if length(seeds) == 1
    elite = copy(get_noise(nt, seeds[1]))
    elite *= ϵ
    return elite
  # Get cached elite
  elseif seeds[1:end-1] in keys(param_cache)
    @inline @inbounds elite = copy(param_cache[seeds[1:end-1]])
    @inline @inbounds elite .+= get_noise(nt, seeds[end]) * ϵ
    return elite
  # Recurse if not cached
  else
    @inline @inbounds elite = reconstruct(param_cache, nt, seeds[1:end-1], ϵ)
    @inline @inbounds elite .+= get_noise(nt, seeds[end]) * ϵ
    return elite
  end
end


# function reconstruct(nt::NoiseTable, x::Vector{<:UInt32}, ϵ::Float32=0.01f0)
#   theta = zeros(Float32, nt.nparams)
#   @inline @views @inbounds theta .+= get_noise(nt, x[1]) ./ 32f0
#   for seed in x[2:end]
#     @inline @inbounds @views noise = get_noise(nt, seed)
#     @simd for i in 1:nt.nparams
#       @views @inbounds theta[i] += noise[i]
#     end
#   end
#   theta .* ϵ
# end



end
