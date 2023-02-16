using StableRNGs
using Flux
using LRUCache
using JLD2
import Base: length

struct ModelInfo
  sizes::Vector{Tuple}
  biases::Vector{Bool}
  re
end
function length(mi::ModelInfo)
  nparams = 0
  for s in mi.sizes
    nparams += prod(s)
  end
  nparams
end


function glorot_normal(rng, dims::Integer...; gain::Real=1)
  std = Float32(gain) * sqrt(1.0f0 / Flux.nfan(dims...)[1])
  randn(rng, Float32, dims...) .* std
end

SeedCache = LRU{Vector{Float32},Dict}

function base_reconstruct(param_cache::SeedCache, nt::NoiseTable, mi::ModelInfo, seeds_and_muts::Vector{Float32}, elite_idxs::Set{Int}, rdc::ReconDataCollector)
  ancestor = gen_params(StableRNG(Int(seeds_and_muts[1])), mi, 1)
  for n in 3:2:length(seeds_and_muts)
    add_noise!(nt, ancestor, UInt32(seeds_and_muts[n]))
  end
  ancestor
end

function reconstruct!(param_cache::SeedCache, nt::NoiseTable, mi::ModelInfo, seeds_and_muts::Vector{Float32}, elite_idxs::Set{Int}, rdc::ReconDataCollector)
  """Reconstruction function that finds the nearest cached ancestor and
  reconstructs all future generations. Creates a new parameter vector if
  no ancestor is found.
  """
  # if we already have node in tree, just perform an access
  cached_ancestor_n = 1
  ancestor::Vector{Float32} = []
  for n in length(seeds_and_muts):-2:3
    if seeds_and_muts[1:n] in keys(param_cache) && 
      haskey(param_cache[seeds_and_muts[1:n]], :params)
      #@inline @inbounds
      ancestor = param_cache[seeds_and_muts[1:n]][:params] |> deepcopy
      cached_ancestor_n = n
      break
    end
    rdc.num_recursions += 1
  end
  if cached_ancestor_n == 1
    ancestor = gen_params(StableRNG(Int(seeds_and_muts[1])), mi, 1)
  end
  for n in cached_ancestor_n+2:2:length(seeds_and_muts)
    add_noise!(nt, ancestor, UInt32(seeds_and_muts[n]))
    if n ∈ elite_idxs
      param_cache[seeds_and_muts[1:n]]= Dict(:params=>deepcopy(ancestor))
    end
  end
  # otherwise, we are reconstructing leaf. ensure all ancestors
  # nodes are still in cache, then return ancestor. Let's do this
  # in sorted order, so each child node will have a cache of a parent.
  # The downside is that the parent might boot out a child from the
  # cache, forcing us to reconstruct the child again. Assuming that 
  # the lower parts of the tree change more than the top, this should
  # not happen to might.
  for eidx in sort(collect(elite_idxs))
    if seeds_and_muts[1:eidx] in keys(param_cache)
      param_cache[seeds_and_muts[1:eidx]][:params]
    else
      param_cache[seeds_and_muts[1:eidx]]= Dict(
           :params=>reconstruct!(param_cache, nt, mi,
              seeds_and_muts[1:eidx], Set{Int}(), rdc))
    end
  end
  return ancestor
end



base_reconstruct(sc::SeedCache, nt::NoiseTable, mi::ModelInfo, ind::Ind, rdc::ReconDataCollector) =
  base_reconstruct(sc, nt, mi, ind.geno, ind.elite_idxs, rdc)
reconstruct!(sc::SeedCache, nt::NoiseTable, mi::ModelInfo, ind::Ind, rdc::ReconDataCollector) =
  reconstruct!(sc, nt, mi, ind.geno, ind.elite_idxs, rdc)

lb(rng, l::Tuple, b::Bool) = b ? zeros(Float32, l) : glorot_normal(rng, l...)
init_params(rng, sizes::Vector{Tuple}, biases::Vector{Bool}) = vcat([lb(rng,l,b)[:] for (l,b) in zip(sizes, biases)]...)
non_init_params(rng, sizes::Vector{Tuple}, biases::Vector{Bool}) =
  vcat(map(x->randn(rng, x...)[:], sizes)...)
function gen_params(rng, lens, biases, gen)
    gen == 1 && return init_params(rng, lens, biases) 
    @assert false
    non_init_params(rng, lens, biases)
end
gen_params(rng, mi::ModelInfo, gen::Int) = gen_params(rng, mi.sizes, mi.biases, gen)


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

