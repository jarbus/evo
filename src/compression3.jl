function find_last_matching_idx(e1::Ind, e2::Ind)
    @assert length(e1.geno) > 0 
    @assert length(e2.geno) > 0
    for i in 1:min(length(e1.geno), length(e2.geno))
        if e1.geno[i] != e2.geno[i]
            return i-1
        end
    end
    return min(length(e1.geno), length(e2.geno))
end


suffix(geno::Geno) = length(geno) > 100 ? @view(geno[end-100:end]) : geno
function suffix_elite(geno::Geno) 
  if length(geno) > 101 
    @view(geno[end-101:end-1])
  elseif length(geno) > 1
    @view(geno[1:end-1])
  else
    geno
  end
end

function compute_elite_idxs!(elites::Vector{Ind})
  """
  For each elite seed, compute all indexes i where elite_seed[1:i]
  is another elite. This can be used for caching results of 
  reconstructions for other elites.
  We also compute and log statistics of the branch ancestors
  """
  elite_idxs = Dict{Geno, EliteIdxs}()
  ancestor_set = Set()
  for e1 in elites
      idxs = Set{Int}()
      for e2 in elites
          last_idx = find_last_matching_idx(e1, e2)
          if last_idx > 0
              push!(idxs, last_idx)
          end
      end
      elite_idxs[suffix(e1.geno)] = idxs
      e1.elite_idxs = idxs
      # for computing idx statistics
      for idx in idxs
        push!(ancestor_set, e1.geno[1:idx])
      end
  end

  num_nodes = length(ancestor_set)
  # compute metrics
  ranges = [maximum(idxs) - minimum(idxs) for (_, idxs) in elite_idxs]
  # num indexes per agent
  num_idxs = [length(idxs) for (_, idxs) in elite_idxs]
  @info "idx_num_unique: |$(num_nodes)|"
  @info "idx_ranges: $(mmms(ranges))"
  @info "idx_num_idxs: $(mmms(num_idxs))"
  elite_idxs
end

function get_elite_idxs(pop::Pop)
    """Apply indicies of parent to each individual 
    """
    eidxs::Dict{Geno, EliteIdxs} = compute_elite_idxs!(pop.elites)
    pop_idxs = Vector{EliteIdxs}(undef, length(pop.inds))

    for (i, ind) in enumerate(pop.inds)
      suf = suffix_elite(ind.geno)
      if haskey(eidxs, suf)
        @inbounds pop_idxs[i] = eidxs[suf]
        continue
      end
      suf = suffix(ind.geno)
      if haskey(eidxs, suf)
        @inbounds pop_idxs[i] = eidxs[suf]
      else
        @inbounds pop_idxs[i] = EliteIdxs()
      end
    end
    pop_idxs
end

function compute_prefixes(pops::Vector{Pop}; k::Int=10)
  elites = vcat([pop.elites for pop in pops]...)
  compute_prefixes(elites; k=k)
end
function compute_prefixes(elites::Vector{Ind}; k::Int=10)
  """Returns the prefixes of elite seeds with the 
  greatest length(prefix)*number_of_elites_with_prefix
  """
  # dict of prefixes => chars_reduced
  chars_reduced = Dict()
  for e1 in elites, e2 in elites
    e1 == e2 && continue
    min_len = min(length(e1.geno), length(e2.geno))
    # go up e1 and e2 until they stop matching
    for i in 1:min_len
      # prefix is the string up until they diverge, or one of them ends
      if e1.geno[i] != e2.geno[i] || i == min_len
        i == 1 && continue # skip if diverge at first char
        idx = e1.geno[i] != e2.geno[i] ? i-1 : min_len
        prefix = e1.geno[1:idx]
        # skip if we've already seen this prefix on a different e1,e2 pair
        haskey(chars_reduced, prefix) && break 
        # once we find a unique prefix, go over all elites to 
        # compute how many chars it might save
        for e3 in elites
          length(e3.geno) < length(prefix) && continue
          if e3.geno[1:length(prefix)] == prefix
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
           rev=true, by=x->x[1])[1:n_prefixes]
  Dict(string(hash(pre))=>pre 
       for (_,pre) in chars_reduced_and_prefix)
end

function compress_elites(pops::Vector{Pop}, prefixes)
  comp_elites::Vector{Vector{RolloutInd}} = []
  for pop in pops
    push!(comp_elites, compress_elites(pop, prefixes))
  end
  comp_elites
end
function compress_pops(pops::Vector{Pop}, prefixes)
  comp_pops::Vector{Vector{RolloutInd}} = []
  for pop in pops
    push!(comp_pops, compress_pop(pop, prefixes))
  end
  comp_pops
end
function compress_pop(pop::Pop, prefixes)
  pop_idxs = get_elite_idxs(pop)
  @assert length(pop_idxs) == length(pop.inds) == pop.size
  # we check prefixes in order of decreasing length in order
  # to maximize the number of characters we can replace
  prefixes_by_len = copy(sort([(length(v), k, v) for (k, v) in prefixes], rev=true))
  comp_pop::Vector{RolloutInd} = []
  geno = Geno()
  for (i,ind) in enumerate(pop.inds)
    found_prefix = false
    for (len, id, prefix) in prefixes_by_len
      if length(ind.geno) >= len && ind.geno[len] == prefix[end] && @view(ind.geno[1:len]) == prefix
        geno = vcat(id, @view(ind.geno[len+1:end]))
        found_prefix = true
        break
      end
    end
    if !found_prefix
      geno = deepcopy(ind.geno)
    end
    push!(comp_pop, RolloutInd(ind.id, geno, pop_idxs[i]))
  end
  comp_pop
end
function compress_elites(pop::Pop, prefixes)
  comp_elites::Vector{RolloutInd} = []
  prefixes_by_len = copy(sort([(length(v), k, v) for (k, v) in prefixes], rev=true))
  for e in pop.elites
    geno::CompGeno = deepcopy(e.geno)
    for (len, id, prefix) in prefixes_by_len
      if length(geno) >= len && geno[1:len] == prefix
        geno = vcat(id, e.geno[len+1:end])|>deepcopy
        break
      end
    end
    push!(comp_elites, RolloutInd(e.id, geno, e.elite_idxs))
  end
  comp_elites
end

function decompress_group(group::Vector{RolloutInd}, prefixes)
  decomp_group::Vector{Ind} = []
  for ro_ind in group
    if typeof(ro_ind.geno[1]) == String
      prefix_seed = prefixes[ro_ind.geno[1]]
      geno = [prefix_seed; ro_ind.geno[2:end]] |> Geno
    else
      geno = deepcopy(ro_ind.geno) |> Geno
    end
    push!(decomp_group, Ind(ro_ind.id, geno, ro_ind.elite_idxs))
  end
  decomp_group
end