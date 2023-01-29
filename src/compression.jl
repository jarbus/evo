function find_last_matching_idx(v1::Vector, v2::Vector)
    for i in 1:2:min(length(v1), length(v2))
        if v1[i] != v2[i]
            i > 1 && return i-2
            i == 1 && return 0
        end
    end
    return min(length(v1), length(v2))
end

function compute_elite_idxs(elites)
    """
    For each elite seed, compute all indexes i where elite_seed[1:i]
    is another elite. This can be used for caching results of 
    reconstructions for other elites
    """
    eseeds = [e[:seeds] for e in elites]
    elite_idxs = Dict()
    for e1 in eseeds
        idxs = Set{Int}()
        for e2 in eseeds
            push!(idxs, find_last_matching_idx(e1, e2))
        end
        elite_idxs[e1] = idxs
    end
    elite_idxs
end

function add_elite_idxs(pop, elites)
  elite_idxs = compute_elite_idxs(elites)
  new_pop = []
  for seed in pop
    if haskey(elite_idxs, seed)
      push!(new_pop, (seed, elite_idxs[seed]))
    elseif haskey(elite_idxs, elite(seed))
      push!(new_pop, (seed, elite_idxs[elite(seed)]))
    else
      push!(new_pop, (seed, Set{Int}()))
    end
  end
  new_pop
end

function compute_prefixes(elites; k::Int=10)
  """Returns the prefixes of elite seeds with the 
  greatest length(prefix)*number_of_elites_with_prefix
  """
  eseeds = [e[:seeds] for e in elites]
  # dict of prefixes => chars_reduced
  chars_reduced = Dict()
  for e1 in eseeds, e2 in eseeds
    e1 == e2 && continue
    min_len = min(length(e1), length(e2))
    # go up e1 and e2 until they stop matching
    for i in 1:2:min_len
      # prefix is the string up until they diverge, or one of them ends
      if e1[i] != e2[i] || i == min_len
        i == 1 && continue # skip if they diverge at the first char
        idx = e1[i] != e2[i] ? i-2 : min_len
        prefix = e1[1:idx]
        # skip if we've already seen this prefix on a different e1,e2 pair
        haskey(chars_reduced, prefix) && break 
        # once we find a unique prefix, go over all elites to 
        # compute how many chars it might save
        for e3 in eseeds
          length(e3) < length(prefix) && continue
          if e3[1:length(prefix)] == prefix
              chars_reduced[prefix] =
                get(chars_reduced, prefix, 0) + length(prefix)
          end
        end
        break
      end
    end
  end
  # sort by number of characters reduced, filter out top k, 
  n_prefixes = min(k, length(chars_reduced))
  chars_reduced_and_prefix = 
      sort([(chrs, prefix) for (prefix, chrs) in chars_reduced],
           rev=true)[1:n_prefixes]
  chars_reduced = [char_and_pre[1] for char_and_pre in chars_reduced_and_prefix]
  @info "Characters reduced with new prefix: $chars_reduced"
  best_prefixes = [char_and_pre[2] for char_and_pre in chars_reduced_and_prefix]
  Dict(string(hash(pre))=>pre for pre in best_prefixes)
end


function compress_pop(pop, elites, prefixes)
    """
    groups: Vector of Tuples t, t[i][1] is identifier , t[i][2] is seed, t[i][3] is a set of idxs
    returns: new_groups, prefix_dict with string(hash(seed)) as key
    """
    new_pop = add_elite_idxs(pop, elites)
    # we check prefixes in order of decreasing length in order to maximize 
    # the number of characters we can replace
    prefixes_by_len = copy(sort([(length(v), k, v) for (k, v) in prefixes], rev=true))
    compressed_pop = []
    for (seed, idx) in new_pop
        for (len, id, prefix) in prefixes_by_len
            if length(seed) >= len && seed[1:len] == prefix
                seed = vcat(id, seed[len+1:end])
                break
            end
        end
        push!(compressed_pop, (seed, idx))
    end
    compressed_pop
end

function decompress_group(group, prefixes)
    new_group = []
    for (id, seeds, elite_idxs) in group
        if typeof(seeds[1]) == String
            prefix_seed = prefixes[seeds[1]]
            seeds = Float32.(vcat(prefix_seed, seeds[2:end]))
        end
        push!(new_group, (id, seeds, elite_idxs))
    end
    new_group
end

