using StableRNGs
using Flux
using LRUCache

struct ModelInfo
  sizes::Vector{Tuple}
  biases::Vector{Bool}
end


function glorot_normal(rng, dims::Integer...; gain::Real=1)
  std = Float32(gain) * sqrt(1.0f0 / Flux.nfan(dims...)[1])
  randn(rng, Float32, dims...) .* std
end

SeedCache = LRU{Vector{Float64},Dict}


function cache_elites!(param_cache::SeedCache, mi::ModelInfo, elites::Vector{<:AbstractDict})
  for elite in elites
    elite[:params] = reconstruct(param_cache, mi, elite[:seeds])
    param_cache[elite[:seeds]] = elite
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
  elseif seeds_and_muts[1:end-2] in keys(param_cache) && haskey(param_cache[seeds_and_muts[1:end-2]], :params)
    @inline @inbounds elite = copy(param_cache[seeds_and_muts[1:end-2]][:params])
    @inline @inbounds elite .+= gen_params(StableRNG(Int(seeds_and_muts[end])), mi, 2) * seeds_and_muts[end-1]
    return elite
  # Recurse if not cached
  else
    length(seeds_and_muts) > 3 && throw("should not reach here")
    @inline @inbounds elite = reconstruct(param_cache, mi, seeds_and_muts[1:end-2])
    @inline @inbounds elite .+= gen_params(StableRNG(Int(seeds_and_muts[end])), mi, 2) * seeds_and_muts[end-1]
    return elite
  end
end


lb(rng, l::Tuple, b::Bool) = b ? zeros(Float32, l) : glorot_normal(rng, l...)
init_params(rng, sizes::Vector{Tuple}, biases::Vector{Bool}) = vcat([lb(rng,l,b)[:] for (l,b) in zip(sizes, biases)]...)
non_init_params(rng, sizes::Vector{Tuple}, biases::Vector{Bool}) =
  vcat(map(x->glorot_normal(rng, x...)[:], sizes)...)
function gen_params(rng, lens, biases, gen)
    gen == 1 && return init_params(rng, lens, biases) 
    non_init_params(rng, lens, biases)
end

function gen_params(rng, mi::ModelInfo, gen::Int)
    gen == 1 && return init_params(rng, mi.sizes, mi.biases) 
    non_init_params(rng, mi.sizes, mi.biases)
end


function ModelInfo(m::Chain)
    lengths = [size(mo) for mo in Flux.params(m)]
    is_bias = [mo isa Vector for mo in Flux.params(m)]
    ModelInfo(lengths, is_bias)
end
