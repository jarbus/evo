using EvoTrade
using Flux
using Test
using StableRNGs

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
#######@testset "test_recachstruction_speed" begin
#######
#######  pop_size = 100
#######  n_elites = 3
#######  model_size = 10_000
#######  param_cache::SeedCache = SeedCache(maxsize=n_elites*3)
#######  m = make_model(:large, (11, 11, 7, 10), 4, lstm=true)
#######  mi = ModelInfo(m)
#######  pop = [rand(1.0f0:1000.0f0, 1) for _ in 1:pop_size]
#######  fitnesses = rand(pop_size)
#######  novelties = rand(pop_size)
#######  bcs = [rand(3) for _ in 1:pop_size]
#######  γ=0.5
#######  pop, elites = create_next_pop(1, param_cache, pop, fitnesses, novelties, bcs, γ, n_elites)
#######  compressed_elites, prefix = compress_elites(param_cache, elites)
#######  cache_elites!(param_cache, mi, compressed_elites, prefix)
#######  reconstruct(param_cache, mi, pop[1])
#######  leave = false
#######  for i in 1:1000
#######      fitnesses = rand(pop_size)
#######      novelties = rand(pop_size)
#######      bcs = [rand(3) for _ in 1:pop_size]
#######      pop, elites = create_next_pop(2, param_cache, pop, fitnesses, novelties, bcs, γ, n_elites)
#######      compressed_elites, prefix = compress_elites(param_cache, elites)
#######      cache_elites!(param_cache, mi, compressed_elites, prefix)
#######      for p in pop
#######        if length(p) >= 100
#######          @btime reconstruct($param_cache, $mi, $p)
#######          leave = true
#######          break
#######        end
#######        leave && break
#######      end
#######      leave && break
#######      # hits = 0
#######   # for j in 1:pop_size
#######   #   theta = reconstruct(param_cache, mi, pop[j])
#######   #   # if rand() < 0.01
#######   #   #   theta2 = reconstruct(nt, pop[j])
#######   #   #   @test all(isapprox.(theta, theta2; atol=0.0001))
#######   #   # end
#######   # end
#######    end
#######  end
#######
#######@testset "test_compression" begin
#######    # Test find_prefix
#######    seeds = [[1],[2],[10, 4, 5,6], [10,4,5,5]]
#######    prefix = EvoTrade.find_prefix(seeds)
#######    @test prefix == [10, 4, 5]
#######    seeds = [[1],[2],[10], [10]]
#######    prefix = EvoTrade.find_prefix(seeds)
#######    @test prefix == [10]
#######    seeds = [[1, 1],[1, 2],[10, 4, 5,6], [10,4,5,5]]
#######    prefix = EvoTrade.find_prefix(seeds)
#######    @test prefix == [10, 4, 5] || prefix == [1]
#######
#######    # Test compress_elites
#######    seeds = [[1],[2],[10, 4, 5,6], [10,4,5,5]]
#######    elites = [Dict{Any, Any}(:seeds=>s) for s in seeds]
#######    sc = Dict()
#######    compressed_elites, prefix = EvoTrade.compress_elites(sc, elites)
#######    @test length(compressed_elites) == 4
#######    @test compressed_elites[1][:seeds] == [1]
#######    @test compressed_elites[2][:seeds] == [2]
#######    @test compressed_elites[3][:seeds] == [:pre, 6]
#######    @test compressed_elites[4][:seeds] == [:pre, 5]
#######
#######    decompressed_elites = EvoTrade.decompress_elites!(compressed_elites, prefix)
#######    for i in eachindex(decompressed_elites)
#######        @test decompressed_elites[i][:seeds] == elites[i][:seeds]
#######    end
#######                   #
#######    # Test decompress_elites
#######end
#######
#######@testset "test_cache_elites" begin
#######
#######  pop_size = 100
#######  n_elites = 3
#######  model_size = 10_000
#######  param_cache::SeedCache = SeedCache(maxsize=n_elites*3)
#######  m = make_model(:large, (11, 11, 7, 10), 4, lstm=true)
#######  mi = ModelInfo(m)
#######  pop = [rand(1.0f0:1000.0f0, 1) for _ in 1:pop_size]
#######  fitnesses = rand(pop_size)
#######  novelties = rand(pop_size)
#######  bcs = [rand(3) for _ in 1:pop_size]
#######  γ=0.5
#######  pop, elites = create_next_pop(1, param_cache, pop, fitnesses, novelties, bcs, γ, n_elites)
#######  compressed_elites, prefix = compress_elites(param_cache, elites)
#######  @test length(compressed_elites) == length(elites)
#######  @test length(prefix) == 1
#######  cache_elites!(param_cache, mi, compressed_elites, prefix)
#######  compressed_elites, prefix = compress_elites(param_cache, elites)
#######  @test length(compressed_elites) == 0
#######  @test length(prefix) == 0
#######  cache_elites!(param_cache, mi, compressed_elites, prefix)
#######  for i in 1:10
#######      fitnesses = rand(pop_size)
#######      novelties = rand(pop_size)
#######      bcs = [rand(3) for _ in 1:pop_size]
#######      pop, elites = create_next_pop(2, param_cache, pop, fitnesses, novelties, bcs, γ, n_elites)
#######      compressed_elites, prefix = compress_elites(param_cache, elites)
#######      cache_elites!(param_cache, mi, compressed_elites, prefix)
#######  end
#######  
#######end


@testset "test_rollout_groups" begin
  pop_size = 100
  n_elites = 3
  model_size = 10_000
  param_cache::SeedCache = SeedCache(maxsize=n_elites*3)
  m = make_model(:large, (11, 11, 7, 10), 4, lstm=true)
  mi = ModelInfo(m)
  pop = [rand(1.0f0:1000.0f0, 1) for _ in 1:pop_size]
  fitnesses = rand(pop_size)
  novelties = rand(pop_size)
  bcs = [rand(3) for _ in 1:pop_size]
  γ=0.5
  groups = create_rollout_groups(pop, 10, 10)
  new_groups = add_elite_idxs_to_groups(groups, [])
  pop, elites = create_next_pop(1, param_cache, pop, fitnesses, novelties, bcs, γ, n_elites)
  cache_elites!(param_cache, mi, elites)
  groups = create_rollout_groups(pop, elites, 10, 10)
  new_groups = add_elite_idxs_to_groups(groups, elites)
  for i in 2:10
      fitnesses = rand(pop_size)
      novelties = rand(pop_size)
      bcs = [rand(3) for _ in 1:pop_size]
      pop, elites = create_next_pop(i, param_cache, pop, fitnesses, novelties, bcs, γ, n_elites)
      prefixes = compute_prefixes(elites)
      cache_elites!(param_cache, mi, elites)
      groups = create_rollout_groups(pop, elites, 1, 1)
      new_groups = add_elite_idxs_to_groups(groups, elites)
      seeds, e_idx = rand(rand(new_groups))[2:3]
      reconstruct(param_cache, mi, seeds, e_idx)
      for ng in new_groups
      # # #   @test length(ng) == 10
        # println(ng[1][3])
      end
  end


end
