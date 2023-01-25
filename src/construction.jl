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

#function cache_elites!(param_cache::SeedCache, mi::ModelInfo, compressed_elites::Vector, prefix::Vector)
#  length(compressed_elites) == 0 && return
#
#  elites = decompress_elites!(compressed_elites, prefix)
#  for elite in elites
#    elite[:params] = reconstruct(param_cache, mi, elite[:seeds])
#  end
#  for elite in elites
#    try
#      param_cache[elite[:seeds]] = elite
#      @assert elite[:seeds] in keys(param_cache)
#    catch
#      @assert elite[:seeds] in keys(param_cache)
#      @assert length(param_cache) <= param_cache.maxsize
#    end
#  end
#end



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
    ModelInfo(lengths, is_bias, re)
end

function save_sc(sc_name::String, sc::SeedCache)
  """Saves a copy of the seed cache without parameters"""
  sc_no_params = SeedCache(maxsize=sc.maxsize)
  for (k,v) in sc
    sc_no_params[k] = Dict(ke=>ve for (ke,ve) in v if ke != :params)
  end
  save(sc_name, Dict("sc"=>sc_no_params))
end


#function find_prefix(seeds::Vector)
#    first_counts = Dict()
#    for seed in seeds
#        if haskey(first_counts, seed[1])
#            first_counts[seed[1]] += 1
#        else
#            first_counts[seed[1]] = 1
#        end
#    end
#    @assert sum(values(first_counts)) == length(seeds)
#    genesis = findmax(first_counts)[2]
#    children_of_genesis = [idx for (idx, s) in enumerate(seeds) if s[1] == genesis]
#    children_lengths = [length(seeds[idx]) for idx in children_of_genesis]
#    max_prefix_size = min(children_lengths...)
#    all_genesis_children_are_same = true
#    prefix = []
#    for i in 1:max_prefix_size
#        current_seeds = [seeds[idx][i] for idx in children_of_genesis] |> unique
#        if length(current_seeds) > 1
#            all_genesis_children_are_same = false
#            break
#        end
#        push!(prefix, current_seeds[1])
#    end
#    prefix
#end
#
#function compress_elites(sc, elites::Vector{<:Dict})
#    # find most common prefixes
#    new_elites = [deepcopy(e) for e in elites if !in(e[:seeds], keys(sc)) && !haskey(e, :params)]
#    if new_elites == []
#      return [], []
#    end
#    prefix = find_prefix([e[:seeds] for e in new_elites])
#    prefix_length = length(prefix)
#    # ignore mutations
#    for i in eachindex(new_elites)
#        if length(new_elites[i][:seeds]) >= prefix_length && new_elites[i][:seeds][1:prefix_length] == prefix
#            new_elites[i][:seeds] = vcat([:pre], new_elites[i][:seeds][prefix_length+1:end])
#        end
#    end
#    new_elites, prefix
#end
#
#function decompress_elites!(elites::Vector{<:AbstractDict}, prefix::Vector)
#    for i in eachindex(elites)
#        if elites[i][:seeds][1] == :pre
#          elites[i][:seeds] = typeof(prefix[1]).(vcat(prefix, elites[i][:seeds][2:end]))
#        end
#        @assert :pre != elites[i][:seeds][1]
#    end
#    elites
#end

