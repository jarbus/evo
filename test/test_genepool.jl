using Random
import EvoTrade: match, MutBinding, accumulate_muts, compute_stats,
                 new_mr, pad_genepool!, make_genepool, add_mutations

sc = SeedCache(maxsize=10)
model = Chain(Dense(1, 1), Dense(1,2))
mi = ModelInfo(model)
nt = NoiseTable(StableRNG(1), length(mi), 1f0)

@testset "match_binding" begin
  geno = rand(Mut, 1)
  mut_match = Mut(rand(Mut).core, missing, MutBinding(0, []))
  mut_nobind = rand(Mut)
  mut_nomatch = Mut(rand(Mut).core,
                    missing,
                    MutBinding(1, [mut_nobind.core]))
  geno = [mut_match]
  @test EvoTrade.match(mut_match, geno)
  # test that trying to match an unbound mutation throws an error
  try
    !EvoTrade.match(mut_nobind, geno)
  catch e
    @test true
  end
  # test that mutation with wrong binding doesn't match
  @test !EvoTrade.match(mut_nomatch, geno)
end

@testset "accumulate_mutations" begin
  # improvements are added
  r_mt = Mut(mi)
  genos = [[Mut(r_mt.core, i+j, r_mt.binding)
           for j in 1:2] for i in 1:10]
  gp = accumulate_muts(genos, 10)
  @test length(gp) == 10
  stats = compute_stats(mi, gp)
  @test stats.num_copied_muts == 10
  @test length(stats.copied_layers_ratios) == 4
  @test length(stats.copied_layers_mrs) == 4
  for g in genos
    @test in(mark_crossover(g[end]), gp)
    @test g[end].crossed_over == false
  end
  for m in gp
    @test m.crossed_over == true
  end
  # mutations that hurt score are not added
  genos = [[Mut(r_mt.core, i-j, r_mt.binding)
           for j in 1:2] for i in 1:10]
  gp = accumulate_muts(genos, 10)
  @test length(gp) == 0
  stats = compute_stats(mi, gp)
  @test stats.num_copied_muts == 0
  @test all(length.(stats.copied_layers_mrs) .== 0)
  pad_genepool!(mi, gp, stats, 10)
  @test length(gp) == 10
  for g in genos
    @test !in(mark_crossover(g[end]), gp)
  end

  # test that only some mutations are added
  genos = [[Mut(r_mt.core, 0, r_mt.binding),
            Mut(r_mt.core, Int(i%2==0), r_mt.binding)]
           for i in 1:10]
  println([genos[i][end].score for i in 1:10])
  gp = accumulate_muts(genos, 5)
  @test length(gp) == 5
  @test mark_crossover(genos[1][end]) ∉ gp
  @test mark_crossover(genos[2][end]) ∈ gp

  # test that highest scores are kept
  genos = [[Mut(r_mt.core, 0, r_mt.binding),
            Mut(r_mt.core, i*Int(i%2==0), r_mt.binding)]
           for i in 1:10]
  gp = accumulate_muts(genos, 2)
  @test length(gp) == 2
  @test genos[2][end].score > genos[2][end-1].score
  @test genos[10][end].score > genos[10][end-1].score
  @test mark_crossover(genos[2][end]) ∉ gp
  @test mark_crossover(genos[10][end]) ∈ gp

  # test that muts from two genos back are used
  genos = [[Mut(r_mt.core, 0, r_mt.binding),
            Mut(r_mt.core, i*Int(i%2==0), r_mt.binding),
            Mut(r_mt.core, 0, r_mt.binding)]
           for i in 1:10]
  gp = accumulate_muts(genos, 2)
  @test length(gp) == 2
  @test mark_crossover(genos[2][2]) ∉ gp
  @test mark_crossover(genos[10][2]) ∈ gp
end

@testset "make_genepool" begin
  genos = [[Mut(Mut(mi).core, i-j, MutBinding(0, []))
           for j in 1:2] for i in 1:10]
  gp = EvoTrade.make_genepool(mi, "1", genos, 20)
  for m in gp
    @test m.crossed_over == false
  end
  genos = add_mutations(gp, genos, 10)
  for g in genos
    @test mark_crossover(g[end]) ∉ gp # test bad muts not in pool
    @test length(g) == 3 # test that new mut is added
    @test g[end].binding.start == 1 # test binding is fresh
    @test g[end].binding.geno == [g[1].core, g[2].core]
    @test mark_crossover(g[end]) ∉ gp
    EvoTrade.update_score!(g, 100f0)
  end
  gp = EvoTrade.make_genepool(mi, "1", genos, 20)
  g_ends = [g[end] for g in genos]
  for m in g_ends
    @test mark_crossover(m) ∈ gp # test good muts added to gene pool
  end
  for m in gp[1:10]
    @test m.crossed_over == true
  end
  for m in gp[11:end]
    @test m.crossed_over == false
  end

end

@testset "create_next_pop" begin
  pop = Pop("1", 10, mi)
  for ind in pop.inds
    ind.fitness = 100f0
  end
  EvoTrade.compute_scores!(pop, 0f0)
  next_pop = EvoTrade.create_next_pop(mi, pop, 0f0, 10)
  @test length(next_pop.inds) == 10

  for ind in next_pop.inds
    ind.fitness = 100f0
  end
  EvoTrade.compute_scores!(next_pop, 0f0)
  next_pop = EvoTrade.create_next_pop(mi, next_pop, 0f0, 2)
  @test length(next_pop.inds) == 10
  @test length(next_pop.elites) == 2
end

@testset "simple_case" begin
  g1 = [Mut(UInt32(1), 1f0)]
  g2 = [Mut(UInt32(2), 1f0)]
  inds = [Ind("1", g1), Ind("2", g2)]
  pop = Pop("1", 2, inds)
  γ = 0.0f0
  EvoTrade.update_score!(pop.inds[1].geno, 100f0)
  pop.inds[1].fitness = 100f0
  EvoTrade.update_score!(pop.inds[2].geno, -100f0)
  pop.inds[2].fitness = -100f0
  gp = EvoTrade.make_genepool(mi, pop)
  @test g1[1] ∉ gp
  @test g2[1] ∉ gp
  new_pop = create_next_pop(mi, pop, γ, 1)
  for ind in new_pop.inds
    length(ind.geno) == 1 && continue
    @test ind.geno[end].binding.start == UInt32(1)
  end
  @test new_pop.inds[1].geno == g1
  @test new_pop.inds[2].geno[1] == g1[1]
  @test length(new_pop.inds[2].geno) == UInt32(2)
  pop = new_pop

  EvoTrade.update_score!(pop.inds[1].geno, -100f0)
  pop.inds[1].fitness = -100f0
  EvoTrade.update_score!(pop.inds[2].geno, 200f0)
  pop.inds[2].fitness = 200f0

  Random.seed!(1)
  gp = EvoTrade.make_genepool(mi, pop)
  @test mark_crossover(pop.inds[1].geno[end]) ∉ gp
  @test mark_crossover(pop.inds[2].geno[end]) ∈ gp
  Random.seed!(1)
  seeds = [g.core.seed for g in gp]
  new_pop = create_next_pop(mi, pop, γ, 1)
  @test new_pop.inds[1].geno == pop.inds[2].geno
  @test new_pop.inds[2].geno[1:2] == pop.inds[2].geno
  @test new_pop.inds[2].geno !== pop.inds[2].geno
  @test new_pop.inds[2].geno[3].core.seed ∈ seeds
  @test length(new_pop.inds[2].geno) == 3
end
