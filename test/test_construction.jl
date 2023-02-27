using EvoTrade
using Flux
using Test
using StableRNGs


@testset "construction" begin

  sc = SeedCache(maxsize=10)
  model = Chain(Dense(1, 1), Dense(1,2))
  mi = ModelInfo(model)
  nt = NoiseTable(StableRNG(1), length(mi), 1f0)
  geno = rand(Mut, 1)
  elite_idxs = EliteIdxs([1])
  rdc = EvoTrade.ReconDataCollector()
  z = construct!(sc, nt, mi, geno, elite_idxs, rdc)
  @test 0 ∈ z # test bias vector
  @test sum(z) != 0 # test there are weights
  pop = Pop("1", 1, mi)
  geno = pop.inds[1].geno
  params = construct!(sc, nt, mi, geno, elite_idxs, rdc)
  # test that new mut with no layers doesn't change params
  push!(geno, rand(Mut))
  new_params = construct!(sc, nt, mi, geno, elite_idxs, rdc)
  @test new_params == params
  # test that new mut with a layer changes some params
  push!(geno, Mut(MutCore(1, 1f0, Set(1))))
  new_params = construct!(sc, nt, mi, geno, elite_idxs, rdc)
  @test new_params != params
  @test new_params[1] != params[1]
  @test new_params[2:end] == params[2:end]

  # test that new mut with all layers changes all params
  push!(geno, Mut(MutCore(1, 1f0, Set([1,2,3,4]))))
  new_new_params = construct!(sc, nt, mi, geno, elite_idxs, rdc)
  @test all(new_new_params .!= new_params)
end

# @testset "reconstruct" begin
#   pop_size = 100
#   n_elites = 3
#   model_size = 10_000
#   param_cache::SeedCache = SeedCache(maxsize=n_elites*3)
#   rdc = EvoTrade.ReconDataCollector()
#   m = make_model(:large, (11, 11, 7, 10), 4, lstm=true)
#   mi = ModelInfo(m)
#   ind1 = Ind("1", [1f0])
#   ind2 = Ind("2", [ind1.geno; [0.5f0]; ind1.geno])
#   ind3 = Ind("3", [ind2.geno; [0.5f0]; ind2.geno])
#   bigind = Ind("4", [1f0 for _ in 1:1001])
#
#   nt = NoiseTable(StableRNG(1), length(mi), 0.5f0)
#   params1 = reconstruct!(param_cache, nt, mi, ind1, rdc)
#   @test rdc.num_recursions == 0
#   params2 = reconstruct!(param_cache, nt, mi, ind2, rdc)
#   @test params1 != params2
#   @test rdc.num_recursions == 1
#   params3 = reconstruct!(param_cache, nt, mi, ind3, rdc)
#   @test params2 != params3
#
#   bigind.elite_idxs = Set([length(bigind.geno)-2])
#   cached_bigparams = reconstruct!(param_cache, nt, mi, bigind, rdc)
#   # @test rdc.num_recursions == 504
#   old_recursions = rdc.num_recursions
#   bigparams = reconstruct!(param_cache, nt, mi, bigind, rdc)
#   @test rdc.num_recursions - old_recursions == 1
#   @test cached_bigparams == bigparams
#   @test bigind.geno[1:201] ∉ keys(param_cache)
#
#   bigind.elite_idxs = Set([length(bigind.geno)-2, 101, 201])
#   bigparams = reconstruct!(param_cache, nt, mi, bigind, rdc)
#   @test bigind.geno[1:101] ∈ keys(param_cache)
#   @test bigind.geno[1:201] ∈ keys(param_cache)
#   @test bigind.geno[1:length(bigind.geno)-2] ∈ keys(param_cache)
#
# end

# @testset "test_new_gen" begin
#     obs_size = (30,32,4, 1)
#     n_actions = 4
#     pop_size = 200
#     m = make_model(:small,
#         obs_size,
#         n_actions,
#         vbn=false,
#         lstm=false)
#     θ, re = Flux.destructure(m)
#     model_size = length(θ)
#     rng = StableRNG(123)
#
#     mi = ModelInfo(m)
#     z = gen_params(rng, mi, 1)
#     # old test to check that biases were zeroed
#     # @test length(findall(x->x==0f0, z)) > 1
#     z = gen_params(rng, mi, 2)
#     @test length(findall(x->x==0f0, z)) <= 1
#     re(z)
#     sc = SeedCache(maxsize=10)
#     a = reconstruct(sc, mi, UInt32.([1,2,3]))
#     b = reconstruct(sc, mi, UInt32.([1,2,3,4]))
#     @test !all(a .== b)
#     a = reconstruct(sc, mi, UInt32.([1]))
#     # @test length(findall(x->x==0f0, z)) > 1
#     # @test length(findall(x->x==0f0, a)) > 1
#     a = reconstruct(sc, mi, UInt32.([1, 2]))
#     @test length(findall(x->x==0f0, a)) <= 1
#
#     seeds = UInt32.([1, 2])
#     a = gen_params(StableRNG(seeds[1]), mi, 1) + gen_params(StableRNG(seeds[2]), mi, 2)
#     b = gen_params(StableRNG(seeds[1]), mi, 1) + gen_params(StableRNG(seeds[2]), mi, 2)
#     @test all(isapprox.(a, b; atol=0.0001))
#     a = reconstruct(sc, mi, UInt32.([1, 2]))
#     b = reconstruct(sc, mi, UInt32.([1, 2]))
#     @test all(isapprox.(a, b; atol=0.0001))
#     a = gen_params(StableRNG(seeds[1]), mi, 1) + gen_params(StableRNG(seeds[2]), mi, 2)
#     b = reconstruct(sc, mi, UInt32.([1, 2]), 1f0)
#     @test all(isapprox.(a, b; atol=0.0001))
#     a = gen_params(StableRNG(seeds[1]), mi, 1) + gen_params(StableRNG(seeds[2]), mi, 2)
#     b = reconstruct(sc, mi, UInt32.([1, 2]), 0.01f0)
#     @test !all(isapprox.(a, b; atol=0.0001))
#
# end

#@testset "test_recachstruction_accuracy" begin
#
#  pop_size = 100
#  n_elites = 3
#  model_size = 10_000
#  param_cache::SeedCache = SeedCache(maxsize=n_elites*3)
#  m = make_model(:large, (11, 11, 7, 10), 4, lstm=true)
#  mi = ModelInfo(m)
#  pop = [rand(1.0:1000.0, 1) for _ in 1:pop_size]
#  fitnesses = rand(pop_size)
#  novelties = rand(pop_size)
#  bcs = [rand(3) for _ in 1:pop_size]
#  γ=0.5
#  pop, elites = create_next_pop(1, param_cache, pop, fitnesses, novelties, bcs, γ, n_elites)
#  cache_elites!(param_cache, mi, elites)
#  reconstruct(param_cache, mi, pop[1])
#
#  for i in 1:1000
#      fitnesses = rand(pop_size)
#      fitnesses[1] = 2
#      novelties = rand(pop_size)
#      bcs = [rand(3) for _ in 1:pop_size]
#      p1_1 = reconstruct(param_cache, mi, pop[1])
#      pop, elites = create_next_pop(2, param_cache, pop, fitnesses, novelties, bcs, γ, n_elites)
#      cache_elites!(param_cache, mi, elites)
#      p1_2 = reconstruct(param_cache, mi, pop[1])
#      p1_3 = reconstruct(param_cache, mi, pop[1])
#      @test p1_1 == p1_2 == p1_3
#      leave = false
#      for p in pop
#        if length(p) >= 100
#          @btime reconstruct($param_cache, $mi, $p)
#          leave = true
#          break
#        end
#        leave && break
#      end
#      leave && break
#      # hits = 0
#   # for j in 1:pop_size
#   #   theta = reconstruct(param_cache, mi, pop[j])
#   #   # if rand() < 0.01
#   #   #   theta2 = reconstruct(nt, pop[j])
#   #   #   @test all(isapprox.(theta, theta2; atol=0.0001))
#   #   # end
#   # end
#  end 
#end
