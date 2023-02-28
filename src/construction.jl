using StableRNGs
using JLD2
import Base: length

param_names(::Conv) = ["conv_w", "conv_b"]
param_names(::Dense) = ["dense_w", "dense_b"]
# asssume all recurrent layers are lstms
param_names(::Flux.Recur) = ["lstm_wi", "lstm_wh", "lstm_b", "lstm_s0", "lstm_s0"]
param_names(c::Chain) = vcat([param_names(l) for l in c.layers]...)
param_names(::Any) = Vector{String}()

function ModelInfo(m::Chain)
    lengths = [size(mo) for mo in Flux.params(m)]
    is_bias = [mo isa Vector for mo in Flux.params(m)]
      _, re = Flux.destructure(m)

    # compute start and end idxs of each layer in param vec
    idxs = cumsum([1; map(prod, lengths)])
    starts_and_ends = [Tuple(idxs[i:i+1]) for i in 1:length(idxs)-1]
    # NOTE: If you are getting an error, it's probably here. 
    # I'm assuming that each unique layer is contained 
    # within a sub-chain, and `m` is a Chain of subchains
    names = param_names(m)
    @assert length(names) == length(lengths)
    ModelInfo(lengths, is_bias, starts_and_ends, names, re)
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


function construct!(param_cache::SeedCache,
                    nt::NoiseTable,
                    mi::ModelInfo,
                    geno::Geno,
                    elite_idxs::EliteIdxs,
                    rdc::ReconDataCollector)
  """Reconstruction function that finds the nearest cached ancestor and
  reconstructs all future generations. Creates a new parameter vector if
  no ancestor is found.
  """
  # if we already have node in tree, just perform an access
  cached_ancestor_n = 1
  ancestor::Vector{Float32} = []
  for n in length(geno):-1:2
    if haskey(param_cache, geno[1:n])
      #@inline @inbounds
      ancestor = param_cache[geno[1:n]] |> deepcopy
      cached_ancestor_n = n
      break
    end
    rdc.num_recursions += 1
  end
  if cached_ancestor_n == 1
    ancestor = gen_params(StableRNG(Int(geno[1].core.seed)), mi, 1)
  end
  for n in cached_ancestor_n+1:length(geno)
    add_noise!(nt, mi, ancestor, geno[n].core)
    if n ∈ elite_idxs
      param_cache[geno[1:n]]= deepcopy(ancestor)
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
    if haskey(param_cache, geno[1:eidx])
      param_cache[geno[1:eidx]]
    else
      param_cache[geno[1:eidx]] = construct!(param_cache, nt, mi,
              geno[1:eidx], EliteIdxs(), rdc)
    end
  end
  return ancestor
end
construct!(sc::SeedCache, nt::NoiseTable, mi::ModelInfo, ind::Ind, rdc::ReconDataCollector) =
  construct!(sc, nt, mi, ind.geno, ind.elite_idxs, rdc)

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



function rm_params(sc::SeedCache)
  """Returns a copy of the seed cache without parameters
  to write the cache to disk without making it too large."""
  sc_no_params = SeedCache(maxsize=sc.maxsize)
  for (k,v) in sc
    sc_no_params[k] = Dict(ke=>ve for (ke,ve) in v if ke != :params)
  end
  sc_no_params
end

