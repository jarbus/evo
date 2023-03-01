
mutate(m::Mut)::Mut = Mut(m, mutate(m.core.mr))
function mutate(mr::MR)
  choice = rand(Float32)
  if choice < 0.4
   return mr * 0.5f0
  elseif 0.4 <= choice < 0.9
   return mr 
 else
   return mr * 2f0
  end
end

function accumulate_muts(genos::Vector{Geno}, size::Int)::GenePool
  """Return a vector of all recent mutations on the genomes
  that have produced a greater score when applied. Make
  copies until size is reached, or cut off worst muts if 
  there are more muts than size.
  """
  muts_and_deltas = []
  num_back = 3 # check end-num_back
  # populate the inital set of mutations
  for g::Geno in genos
    for i in max(length(g)-num_back, 1):length(g)
      if i > 1 && g[i].score > g[i-1].score
        Δscore = g[i].score - g[i-1].score
        push!(muts_and_deltas, (Δscore, mark_crossover(g[i])))
      end
    end
  end
  # sort the mutations by score
  sort!(muts_and_deltas, by=x->x[1], rev=true)

  # cut of worst mutations if there are too many
  if length(muts_and_deltas) >= size
    muts_and_deltas = muts_and_deltas[1:size]
  # make copies of random mutations if not enough
  elseif 1 < length(muts_and_deltas) < size
    while length(muts_and_deltas) < size
      push!(muts_and_deltas, rand(muts_and_deltas))
    end
  # Otherwise, return an empty gene pool
  else
    return Vector{Mut}()
  end
  muts = [m[2] for m in muts_and_deltas]
  @assert length(muts) == size
  muts
end

function compute_stats(mi::ModelInfo, gp::GenePool)::GenePoolStatistics
  """Compute statistics on the accumulated gene pool. We need this to
  figure out how many new mutations to add, and what mr they should use."""
  num_copied_muts = 0
  copied_layers_counts = zeros(Float32, length(mi.sizes))
  copied_layers_mrs = [Vector{Float32}() for _ in 1:length(mi.sizes)]
  seen_muts = Set{Mut}()
  for mut in gp
    ismissing(mut.score) && continue
    mut in seen_muts && continue
    push!(seen_muts, mut)
    num_copied_muts += 1
    for layer in mut.core.layers
      push!(copied_layers_mrs[layer], mut.core.mr)
      copied_layers_counts[layer] += 1
    end
  end
  copied_ratios = copied_layers_counts ./ sum(copied_layers_counts)
  GenePoolStatistics(num_copied_muts, copied_ratios, copied_layers_mrs)
end

# function make_new_mutation(mi::ModelInfo,
#                           stats::GenePoolStatistics)::Mut
#   """Create a new mutation, informed by gpool statistics"""
#     layer = rand(1:length(mi.sizes))
#     if 0 == length(stats.copied_layers_mrs[layer])
#       mr = new_mr()
#     else
#       mr = rand(stats.copied_layers_mrs[layer])
#     end
#     new_core = MutCore(rand(UInt32), mr, Set([layer]))
#     Mut(new_core, missing, MutBinding(missing, []))
# end

function make_new_mutation(mi::ModelInfo,
                          stats::GenePoolStatistics)::Mut
  """Create a new mutation, UNinformed by gpool statistics"""
  Mut(mi)
end

function pad_genepool!(mi::ModelInfo, 
                       gp::GenePool,
                       stats::GenePoolStatistics,
                       num::Int)
  """Pad the gene pool with random mutations"""

  for _ in 1:num
    push!(gp, make_new_mutation(mi, stats))
  end
end


match(mut::Mut, geno::Geno) = match(mut.binding, geno)
function match(bind::MutBinding, geno::Geno)::Bool
  """Return true if the mutation binding matches the genome"""
  ismissing(bind.start)     && return true
  length(geno) < bind.start && return false
  @simd for i in 1:length(bind.geno)
    @assert 1 <= i+bind.start-1 <= length(geno)
    if geno[bind.start+i-1].core != bind.geno[i]
      return false
    end
  end
  true
end

function log_genepool_stats(mi::ModelInfo, id::String, stats::GenePoolStatistics)
  @info "$(id)_stats.num_copied_muts: |$(stats.num_copied_muts)|"
  @info "$(id)_stats.copied_ratios: $(stats.copied_layers_ratios)"
  mean_mrs = filter(!isnan, mean.(stats.copied_layers_mrs))
  @info "$(id)_stats.copied_mrs: $(mmms(mean_mrs))"
  ratios = Dict{String, Float32}()
  for i in eachindex(stats.copied_layers_ratios)
    ratios[mi.names[i]] = get(ratios, mi.names[i], 0f0) + stats.copied_layers_ratios[i]
  end
  for (name, ratio) in ratios
    @info "$(id)_stats.copied_ratios.$name: $ratio"
  end
  if stats.num_copied_muts > 0
    @assert sum(values(ratios)) ≈ 1f0
  end
end

make_genepool(model_info::ModelInfo, pop::Pop) =
  make_genepool(model_info, pop.id, genos(pop), pop.size)
function make_genepool(mi::ModelInfo, id::String, genos::Vector{Geno}, size::Int)::GenePool
  gp = accumulate_muts(genos, Int(size/2))
  stats = compute_stats(mi, gp)
  log_genepool_stats(mi, id, stats)
  pad_genepool!(mi, gp, stats, size - length(gp))
  @assert length(gp) == size
  gp
end

function create_binding(geno::Geno)::MutBinding
  """Create a binding for a mutation"""
  b_start = max(1,length(geno)-5)
  b_end = min(length(geno), b_start+2)
  MutBinding(b_start, [m.core for m in geno[b_start:b_end]])
end

function update_score!(geno::Geno, score::Float32)
  """Update the score of the last mutation in the genome"""
  geno[end] = mark_score(geno[end], score)
end

function add_mutation(geno::Geno, mut::Mut)
  """Adds a mutation to a genome, creating a binding if
  the binding is missing. Assumes mutation is a match."""
  new_geno = deepcopy(geno)
  # Bind mutation if not bound
  if ismissing(mut.binding.start)
    binding = create_binding(new_geno)
    bound_mut = Mut(mut, binding)
    mutated_mut = mutate(bound_mut)
  else
    mutated_mut = mutate(mut)
    @assert mutated_mut.crossed_over
  end
  push!(new_geno, mutated_mut)
  new_geno
end

function add_mutations(gp::GenePool,
                        genos::Vector{Geno},
                        n::Int)
  """Add a mutation to each genome. Also return a
  vec of bools indicating if the mutation was a crossover"""
  new_genos = Vector{Geno}(undef, n)
  muts = collect(gp)
  for i in 1:n
    geno = rand(genos)
    added = false
    for m in randperm(length(muts))
      if match(muts[m], geno)
        new_genos[i] = add_mutation(geno, muts[m])
        added = true
        break
      end
    end
    @assert added == true
  end
  new_genos
end

log_improvements(pops::Vector{Pop}) = [log_improvements(p) for p in pops]
log_improvements(p::Pop) = log_improvements(p.id, genos(p))
function log_improvements(id::String, genos::Vector{Geno})
  """To be applied once new mutations have an associated score"""
  num_crossovers = sum(g[end].crossed_over for g in genos)
  crossover_deltas = Float32[]
  non_crossover_deltas = Float32[]
  for geno in genos
    if geno[end].crossed_over && length(geno) > 1
      crossover_deltas = [crossover_deltas; geno[end].score - geno[end-1].score]
    elseif length(geno) > 1
      non_crossover_deltas = [non_crossover_deltas; geno[end].score - geno[end-1].score]
    end
  end
  @info "$(id)_num_crossovers: |$num_crossovers|"
  @info "$(id)_crossover_deltas: $(mmms(crossover_deltas))"
  @info "$(id)_non_crossover_deltas: $(mmms(non_crossover_deltas))"
end
