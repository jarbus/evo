using EvoTrade
using Flux
using Test
using StableRNGs

@testset "test_new_gen" begin
    obs_size = (30,32,4, 1)
    n_actions = 4
    pop_size = 200
    m = make_model(:small,
        obs_size,
        n_actions,
        vbn=false,
        lstm=false)
    θ, re = Flux.destructure(m)
    model_size = length(θ)
    rng = StableRNG(123)

    mi = ModelInfo(m)
    z = gen_params(rng, mi, 1)
    # old test to check that biases were zeroed
    # @test length(findall(x->x==0f0, z)) > 1
    z = gen_params(rng, mi, 2)
    @test length(findall(x->x==0f0, z)) <= 1
    re(z)
    sc = SeedCache(maxsize=10)
    a = reconstruct(sc, mi, UInt32.([1,2,3]))
    b = reconstruct(sc, mi, UInt32.([1,2,3,4]))
    @test !all(a .== b)
    a = reconstruct(sc, mi, UInt32.([1]))
    # @test length(findall(x->x==0f0, z)) > 1
    # @test length(findall(x->x==0f0, a)) > 1
    a = reconstruct(sc, mi, UInt32.([1, 2]))
    @test length(findall(x->x==0f0, a)) <= 1

    seeds = UInt32.([1, 2])
    a = gen_params(StableRNG(seeds[1]), mi, 1) + gen_params(StableRNG(seeds[2]), mi, 2)
    b = gen_params(StableRNG(seeds[1]), mi, 1) + gen_params(StableRNG(seeds[2]), mi, 2)
    @test all(isapprox.(a, b; atol=0.0001))
    a = reconstruct(sc, mi, UInt32.([1, 2]))
    b = reconstruct(sc, mi, UInt32.([1, 2]))
    @test all(isapprox.(a, b; atol=0.0001))
    a = gen_params(StableRNG(seeds[1]), mi, 1) + gen_params(StableRNG(seeds[2]), mi, 2)
    b = reconstruct(sc, mi, UInt32.([1, 2]), 1f0)
    @test all(isapprox.(a, b; atol=0.0001))
    a = gen_params(StableRNG(seeds[1]), mi, 1) + gen_params(StableRNG(seeds[2]), mi, 2)
    b = reconstruct(sc, mi, UInt32.([1, 2]), 0.01f0)
    @test !all(isapprox.(a, b; atol=0.0001))

end

@testset "test_NoiseTable" begin
  rng = StableRNG(123)
  nt = NoiseTable(rng, 2, 4, 0.1f0)
  @test get_noise(nt, 1) == get_noise(nt, 1)
  @test get_noise(nt, 1) != get_noise(nt, 2)
end


@testset "test_nt_reconstruct" begin
  nt = NoiseTable(StableRNG(123), 2_460_000, 20_000, 0.1f0)
  seeds = UInt32.([rand([3,4,5]) for _ in 1:30])
  θ1 =  EvoTrade.NoiseTables.reconstruct(nt, seeds)
  θ2 =  EvoTrade.NoiseTables.reconstruct(nt, seeds)
  @test θ1 == θ2
end

# @testset "test_recachstruction_accuracy" begin


#   function create_next_gen(pop::Vector{Vector{UInt32}}, num_elites::Int=60)
#     elites = pop[1:num_elites]
#     next_gen = [vcat(rand(elites), UInt32.(rand(1:length(pop), 1))) for _ in 1:length(pop)]
#     next_gen
#   end

#   pop_size = 1_000
#   n_elites = 3
#   model_size = 10_000
#   param_cache::SeedCache = SeedCache(maxsize=n_elites*2)
#   nt = NoiseTable(StableRNG(123), model_size, pop_size, 1f0)

#   pop = [UInt32.(rand(1:pop_size, 1)) for _ in 1:pop_size]
#   reconstruct(param_cache, nt, pop[1])
#   for i in 1:50
#     cache_elites!(param_cache, nt, pop[1:n_elites], 0.01f0)
#     pop = create_next_gen(pop, n_elites)
#     hits = 0
#     for j in 1:pop_size
#       theta = reconstruct(param_cache, nt, pop[j])
#       if rand() < 0.01
#         theta2 = reconstruct(nt, pop[j])
#         @test all(isapprox.(theta, theta2; atol=0.0001))
#       end
#     end
#   end
# end
