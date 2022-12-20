module NoiseTables
export NoiseTable, compute_grad, get_noise, reconstruct, SeedCache, cache_elites!
export ModelInfo, gen_params
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

struct ModelInfo
  lengths::Vector{Int64}
  biases::Vector{Bool}
end

NoiseTable(rng::StableRNGs.LehmerRNG, nparams::Int, pop_size::Int, σ::Float32) = NoiseTable(rng, Flux.glorot_normal(rng, nparams + pop_size), nparams, pop_size, σ)
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
    @inline @views @inbounds theta .+= get_noise(nt, seed) * ϵ
  end
  theta
end

SeedCache = LRU{Vector{Float64},Dict}

function cache_elites!(param_cache::SeedCache, nt::NoiseTable, elites::Vector{Vector{UInt32}}, ϵ::Float32=0.01)
  """On each processor, cache the parameters for all elites used to create the next 
  generation this makes reconstruction per generation O(T), where T is the truncation size
  """
  for elite in elites
    @inbounds param_cache[elite] = reconstruct(param_cache, nt, elite, ϵ)
  end
end


function cache_elites!(param_cache::SeedCache, mi::ModelInfo, elites::Vector{Vector{UInt32}}, ϵ::Float32=0.01)
  """On each processor, cache the parameters for all elites used to create the next 
  generation this makes reconstruction per generation O(T), where T is the truncation size
  """
  for elite in elites
    @inbounds param_cache[elite] = reconstruct(param_cache, mi, elite, ϵ)
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

function reconstruct(param_cache::SeedCache, mi::ModelInfo, seeds::Vector{UInt32}, ϵ::Float32=0.01f0)
  """Reconstruction function that finds the nearest cached ancestor and
  reconstructs all future generations. Creates a new parameter vector if
  no ancestor is found.
  """
  if length(seeds) == 1
    elite = gen_params(StableRNG(seeds[1]), mi, 1)
    # elite *= ϵ
    return elite
  # Get cached elite
  elseif seeds[1:end-1] in keys(param_cache)
    @inline @inbounds elite = copy(param_cache[seeds[1:end-1]])
    @inline @inbounds elite .+= gen_params(StableRNG(seeds[end]), mi, 2) * ϵ
    return elite
  # Recurse if not cached
  else
    @inline @inbounds elite = reconstruct(param_cache, mi, seeds[1:end-1], ϵ)
    @inline @inbounds elite .+= gen_params(StableRNG(seeds[end]), mi, 2) * ϵ
    return elite
  end
end

function cache_elites!(param_cache::SeedCache, mi::ModelInfo, elites::Vector{<:AbstractDict})
  for elite in elites
    elite[:params] = reconstruct(param_cache, mi, elite[:seeds])
    @inbounds param_cache[elite[:seeds]] = elite
  end
end


function reconstruct(param_cache::SeedCache, mi::ModelInfo, seeds_and_muts::Vector{Float64})
  """Reconstruction function that finds the nearest cached ancestor and
  reconstructs all future generations. Creates a new parameter vector if
  no ancestor is found.
  """
  if length(seeds_and_muts) == 1
    elite = gen_params(StableRNG(Int(seeds_and_muts[1])), mi, 1)
    # elite *= ϵ
    return elite
  @assert isodd(length(seeds_and_muts))
  # Get cached elite
  elseif seeds_and_muts[1:end-2] in keys(param_cache)
    @inline @inbounds elite = copy(param_cache[seeds_and_muts[1:end-2]][:params])
    @inline @inbounds elite .+= gen_params(StableRNG(Int(seeds_and_muts[end])), mi, 2) * seeds_and_muts[end-1]
    return elite
  # Recurse if not cached
  else
    @inline @inbounds elite = reconstruct(param_cache, mi, seeds_and_muts[1:end-2])
    @inline @inbounds elite .+= gen_params(StableRNG(Int(seeds_and_muts[end])), mi, 2) * seeds_and_muts[end-1]
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

lb(rng, l::Int64, b::Bool) = b ? zeros(Float32, l) : Flux.glorot_normal(rng, l)
init_params(rng, lens::Vector{Int64}, biases::Vector{Bool}) =
    vcat([lb(rng, l,b) for (l,b) in zip(lens, biases)]...)
non_init_params(rng, lens::Vector{Int64}, biases::Vector{Bool}) =
  vcat(map(x->Flux.glorot_normal(rng, x), lens)...)
function gen_params(rng, lens, biases, gen)
    gen == 1 && return init_params(rng, lens, biases) 
    non_init_params(rng, lens, biases)
end

function gen_params(rng, mi::ModelInfo, gen::Int)
    gen == 1 && return non_init_params(rng, mi.lengths, mi.biases) 
    non_init_params(rng, mi.lengths, mi.biases)
end


function ModelInfo(m::Chain)
    lengths = [length(mo) for mo in Flux.params(m)]
    is_bias = [mo isa Vector for mo in Flux.params(m)]
    ModelInfo(lengths, is_bias)
end




end
