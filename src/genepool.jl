
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

function accumulate_muts(genos::Vector{Geno})::GenePool
  """Return a set of all recent mutations on the genomes
  that have produced a greater score when applied
  """
  muts = GenePool()
  num_back = 3 # check end-num_back
  for g::Geno in genos
    for i in max(length(g)-num_back, 1):length(g)
      if i > 1 && g[i].score > g[i-1].score
        push!(muts, g[i])
      end
    end
  end
  muts
end

function compute_stats(mi::ModelInfo, gp::GenePool)::GenePoolStatistics
  """Compute statistics on the accumulated gene pool. We need this to
  figure out how many new mutations to add, and what mr they should use."""
  num_copied_muts = 0
  copied_layers_counts = zeros(Float32, length(mi.sizes))
  copied_layers_mrs = [Vector{Float32}() for _ in 1:length(mi.sizes)]
  for mut in gp
    ismissing(mut.score) && continue
    num_copied_muts += 1
    for layer in mut.core.layers
      push!(copied_layers_mrs[layer], mut.core.mr)
      copied_layers_counts[layer] += 1
    end
  end
  copied_ratios = copied_layers_counts ./ sum(copied_layers_counts)
  GenePoolStatistics(num_copied_muts, copied_ratios, copied_layers_mrs)
end

function make_new_mutation(mi::ModelInfo,
                          stats::GenePoolStatistics)::Mut
  """Create a new mutation"""
    layer = rand(1:length(mi.sizes))
    if 0 == length(stats.copied_layers_mrs[layer])
      mr = new_mr()
    else
      mr = rand(stats.copied_layers_mrs[layer])
    end
    new_core = MutCore(rand(UInt32), mr, Set([layer]))
    Mut(new_core, missing, MutBinding(missing, []))
end

function pad_genepool!(mi::ModelInfo, 
                       gp::GenePool,
                       stats::GenePoolStatistics,
                       size::Int)
  """Pad the gene pool with random mutations"""
  while length(gp) < size
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
    if geno[bind.start+i-1] != bind.geno[i]
      return false
    end
  end
  true
end

make_genepool(model_info::ModelInfo, pop::Pop) =
  make_genepool(model_info,
                [ind.geno for ind in pop.inds],
                pop.size)
function make_genepool(mi::ModelInfo, genos::Vector{Geno}, size::Int)::GenePool
  gp = accumulate_muts(genos)
  stats = compute_stats(mi, gp)
  @info "stats.num_copied_muts: $(stats.num_copied_muts)"
  @info "stats.copied_ratios: $(stats.copied_layers_ratios)"
  mean_mrs = filter(!isnan, mean.(stats.copied_layers_mrs))
  @info "stats.copied_mrs: $(mmms(mean_mrs))"
  # we want to pad to have at least 50% new mutations
  final_size = max(size, 2*length(gp))
  pad_genepool!(mi, gp, stats, final_size)
  gp
end

function create_binding(mut::Mut, geno::Geno)::MutBinding
  """Create a binding for a mutation"""
  start = max(1,length(geno)-3)
  MutBinding(start, [m.core for m in geno[start:end]])
end

function update_score!(geno::Geno, score::Float32)
  """Update the score of the last mutation in the genome"""
  geno[end] = Mut(geno[end].core, score, geno[end].binding)
end

function add_mutation(geno::Geno, mut::Mut)
  """Adds a mutation to a genome, creating a binding if
  the binding is missing. Assumes mutation is a match."""
  new_geno = deepcopy(geno)
  # Bind mutation if not bound
  if ismissing(mut.binding.start)
    binding = create_binding(mut, new_geno)
    bound_mut = Mut(mut.core, mut.score, binding)
    mutated_mut = mutate(bound_mut)
  else
    mutated_mut = mutate(mut)
  end
  push!(new_geno, mutated_mut)
  new_geno
end

function add_mutations(gp::GenePool,
                        genos::Vector{Geno},
                        n::Int)
  new_genos = Vector{Geno}(undef, n)
  muts = collect(gp)
  shuffle!(muts)
  for i in 1:n
    geno = rand(genos)
    added = false
    for mut in muts
      if match(mut, geno)
        new_genos[i] = add_mutation(geno, mut)
        added = true
        break
      end
    end
    @assert added
    shuffle!(muts)
  end
  new_genos
end
