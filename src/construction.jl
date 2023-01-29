using StableRNGs
using Flux
using LRUCache
using JLD2

struct ModelInfo
  sizes::Vector{Tuple}
  biases::Vector{Bool}
  re
end


function glorot_normal(rng, dims::Integer...; gain::Real=1)
  std = Float32(gain) * sqrt(1.0f0 / Flux.nfan(dims...)[1])
  randn(rng, Float32, dims...) .* std
end

SeedCache = LRU{Vector{Float32},Dict}

function cache_elites!(param_cache::SeedCache, mi::ModelInfo, elites::Vector{<:AbstractDict})
  #for elite in elites
  #  elite[:params] = reconstruct(param_cache, mi, elite[:seeds])
  #end
  for elite in elites
    try
      param_cache[elite[:seeds]] = elite
    catch
      @assert elite[:seeds] in keys(param_cache)
      @assert length(param_cache) <= param_cache.maxsize
    end
  end
end

function reconstruct(param_cache::SeedCache, mi::ModelInfo, seeds_and_muts::Vector, elite_idxs::Set{Int}=Set{Int}())
  """Reconstruction function that finds the nearest cached ancestor and
  reconstructs all future generations. Creates a new parameter vector if
  no ancestor is found.
  """
  @assert isodd(length(seeds_and_muts))
  if length(seeds_and_muts) == 1
    elite = gen_params(StableRNG(Int(seeds_and_muts[1])), mi, 1)
    return elite
  # elite that was directly copied over 
  elseif seeds_and_muts in keys(param_cache) && haskey(param_cache[seeds_and_muts], :params)
    return deepcopy(param_cache[seeds_and_muts][:params])
  # Get cached parent
  elseif seeds_and_muts[1:end-2] in keys(param_cache) && haskey(param_cache[seeds_and_muts[1:end-2]], :params)
    @inline @inbounds elite = param_cache[seeds_and_muts[1:end-2]][:params] |> deepcopy
  # Recurse if not cached
  else
    @inline @inbounds elite = reconstruct(param_cache, mi, seeds_and_muts[1:end-2], elite_idxs)
  end
  @inline @inbounds elite .+= gen_params(StableRNG(Int(seeds_and_muts[end])), mi, 2) * seeds_and_muts[end-1]
  if length(seeds_and_muts) ∈ elite_idxs
    param_cache[seeds_and_muts] = Dict(:params => deepcopy(elite))
  end
  return elite
end
reconstruct(sc::SeedCache, mi::ModelInfo, ind::Ind) =
  reconstruct(sc, mi, ind.geno, ind.elite_idxs)


lb(rng, l::Tuple, b::Bool) = b ? zeros(Float32, l) : glorot_normal(rng, l...)
init_params(rng, sizes::Vector{Tuple}, biases::Vector{Bool}) = vcat([lb(rng,l,b)[:] for (l,b) in zip(sizes, biases)]...)
non_init_params(rng, sizes::Vector{Tuple}, biases::Vector{Bool}) =
  vcat(map(x->randn(rng, x...)[:], sizes)...)
function gen_params(rng, lens, biases, gen)
    gen == 1 && return init_params(rng, lens, biases) 
    non_init_params(rng, lens, biases)
end

function gen_params(rng, mi::ModelInfo, gen::Int)
    gen == 1 && return init_params(rng, mi.sizes, mi.biases) 
    non_init_params(rng, mi.sizes, mi.biases)
end


function ModelInfo(m::Chain, re=nothing)
    lengths = [size(mo) for mo in Flux.params(m)]
    is_bias = [mo isa Vector for mo in Flux.params(m)]
    if !isnothing(re)
      _, re = Flux.destructure(m)
    end
    ModelInfo(lengths, is_bias, re)
end

function rm_params(sc::SeedCache)
  """Returns a copy of the seed cache without parameters
  to write the cache to disk without making it too large."""
  sc_no_params = SeedCache(maxsize=sc.maxsize)
  for (k,v) in sc
    sc_no_params[k] = Dict(ke=>ve for (ke,ve) in v if ke != :params)
  end
  sc_no_params
end

