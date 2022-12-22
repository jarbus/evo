using EvoTrade
using Test
using Flux

elite(x) = length(x) > 2 ? x[1:end-2] : x
mr(x) = length(x) > 1 ? x[end-1] : 10.0 ^ rand([-1,-2,-3,-4,-5])

@testset "test_reconstruct_with_σ" begin
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
    p1 = [1, 0.1, 2, 0.2, 3]
    p2 = [1, 0.1, 2, 0.1, 3]
    sc = SeedCache(maxsize=10)
    a = reconstruct(sc, mi, p1)
    b = reconstruct(sc, mi, p2)
    c = reconstruct(sc, mi, p1)
    @test !all(a .== b)
    @test all(a .== c)
end


@testset "test_multi_gen_γ=0.5" begin
    obs_size = (30,32,4, 1)
    n_actions = 4
    pop_size = 6
    m = make_model(:small,
        obs_size,
        n_actions,
        vbn=false,
        lstm=false)

    sc = SeedCache(maxsize=4)
    mi = ModelInfo(m)
    p1 = [1.]
    p2 = [2.]
    pop = [p1, p2, [0.0], [0.0], [0.0], [0.0]]
    fitnesses = [2., 1., 0., 0., 0., 0.]
    novelties = [1., 2., 0., 0., 0., 0.]
    bcs = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
    γ = 0.5
    pop, elites = create_next_pop(1, sc, pop, fitnesses, novelties, bcs, γ, 2)
    cache_elites!(sc, mi, elites)
    @test elites[1][:seeds] == [1.]
    @test elites[2][:seeds] == [2.]
    @test elites[1][:seeds] ∈ keys(sc)
    @test elites[2][:seeds] ∈ keys(sc)
    @test elites[2][:seeds] == [2.]
    @test elites[1][:seeds] != [0.0] != elites[2][:seeds]
    for g in 2:10
        bcs[2] = bcs[2] .+ 1 # make sure that the second seed is always novel
        fitnesses .+= fitnesses
        novelties .+= novelties
        next_pop, elites = create_next_pop(g, sc, pop, fitnesses, novelties, bcs, γ, 3)
        best_f_mr = mr(elites[1][:seeds])
        best_n_mr = mr(elites[2][:seeds])
        cache_elites!(sc, mi, elites)

        @test p1 == next_pop[1] == elites[1][:seeds]
        @test next_pop[1] == elites[1][:seeds]
        @test next_pop[2][1:end-2] == elites[1][:seeds] || next_pop[2][1:end-2] == elites[2][:seeds]
        @test next_pop[3][1:end-2] == elites[1][:seeds] || next_pop[3][1:end-2] == elites[2][:seeds]
        @test next_pop[4][1:end-2] == elites[1][:seeds] || next_pop[4][1:end-2] == elites[2][:seeds]
        @test next_pop[5][1:end-2] == elites[3][:seeds]
        @test next_pop[6][1:end-2] == elites[3][:seeds]

        pop = next_pop
    end
end

@testset "test_multi_gen_γ={0.001,0.999}" begin
    # assure there are no crashes at weird γ values
    obs_size = (30,32,4, 1)
    n_actions = 4
    pop_size = 10
    n_elites = 3
    m = make_model(:small,
        obs_size,
        n_actions,
        vbn=false,
        lstm=false)
    for γ in [0.001, 0.90]
        sc = SeedCache(maxsize=2*n_elites)
        mi = ModelInfo(m)
        p1 = [1.]
        p2 = [2.]
        pop = [p1, p2, [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
        fitnesses = [2., 1., 0., 0., 0., 0., 0.0, 0.0, 0.0, 0.0]
        novelties = [1., 2., 0., 0., 0., 0., 0.0, 0.0, 0.0, 0.0]
        bcs = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
        pop, elites = create_next_pop(1, sc, pop, fitnesses, novelties, bcs, γ, 2)
        cache_elites!(sc, mi, elites)
        for g in 2:5
            fitnesses .+= fitnesses
            novelties .+= novelties
            next_pop, elites = create_next_pop(g, sc, pop, fitnesses, novelties, bcs, γ, 3)
            cache_elites!(sc, mi, elites)

            @test p1 == next_pop[1] == elites[1][:seeds]
            @test next_pop[1] == elites[1][:seeds]
            # @test next_pop[2][1:end-2] == elites[1][:seeds] || next_pop[2][1:end-2] == elites[2][:seeds]
            # @test next_pop[3][1:end-2] == elites[1][:seeds] || next_pop[3][1:end-2] == elites[2][:seeds]
            # @test next_pop[4][1:end-2] == elites[1][:seeds] || next_pop[4][1:end-2] == elites[2][:seeds]
            # @test next_pop[5][1:end-2] == elites[3][:seeds]
            # @test next_pop[6][1:end-2] == elites[3][:seeds]
            pop = next_pop
        end
    end
end

@testset "test_mutation" begin
    for i in rand(10) * 0.1
        m = M(i)
        @test 0.5i <= m <= min(2i, 0.1)
    end
end

@testset "test_multigen_γ={0,1}" begin
   obs_size = (30,32,4, 1)
   n_actions = 4
   pop_size = 6
   m = make_model(:small,
       obs_size,
       n_actions,
       vbn=false,
       lstm=false)

   mi = ModelInfo(m)
   p1 = [1.]
   p2 = [2.]
   fitnesses = [2., 1., 1.5, 0., 0., 0.]
   novelties = [1., 2., 0., 0., 0., 0.]
   bcs = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
   sc = SeedCache(maxsize=4)
   γ = 0.0
   pop = [p1, p2, [0.0], [0.0], [0.0], [0.0]]
   pop, elites = create_next_pop(1, sc, pop, fitnesses, novelties, bcs, γ, 2)
   cache_elites!(sc, mi, elites)
   @test elites[1][:seeds] == p1
   @test elites[2][:seeds] == [0.0]
   @test length(findall(x->x==p1, pop)) == 1
   for g in 2:5
       bcs[2] = bcs[2] .+ 1 # make sure that the second seed is always novel
       pop, elites = create_next_pop(g, sc, pop, fitnesses, novelties, bcs, γ, 2)
       f_mr = mr(elites[1][:seeds])
       f_mr2 = mr(elites[2][:seeds])
       cache_elites!(sc, mi, elites)
   end
   sc = SeedCache(maxsize=4)
   γ = 0.9
   pop = [p1, p2, [0.0], [0.0], [0.0], [0.0]]
   pop, elites = create_next_pop(1, sc, pop, fitnesses, novelties, bcs, γ, 3)
   cache_elites!(sc, mi, elites)
   @test elites[1][:seeds] == p1
   @test elites[2][:seeds] == p2
   @test elites[3][:seeds] == p1
   @test length(findall(x->x==p1, pop)) == 1
   for g in 2:5
       bcs[2] = bcs[2] .+ 1 # make sure that the second seed is always novel
       pop, elites = create_next_pop(g, sc, pop, fitnesses, novelties, bcs, γ, 2)
       n_mr = mr(elites[1][:seeds])
       n_mr2 = mr(elites[2][:seeds])
       cache_elites!(sc, mi, elites)
   end
end


@testset "test_large_pop" begin
    obs_size = (30,32,4, 1)
    n_actions = 4
    pop_size = 2000
    num_elites = 15
    m = make_model(:small,
        obs_size,
        n_actions,
        vbn=false,
        lstm=false)

    sc = SeedCache(maxsize=num_elites*2)
    mi = ModelInfo(m)
    pop = [Vector{Float64}([rand(UInt32)]) for i in 1:pop_size]
    γ = 0.5 

    for g in 1:1
        fitnesses = rand(pop_size)
        novelties = rand(pop_size)
        bcs = [[rand()] for i in 1:pop_size]
        pop, elites = create_next_pop(g, sc, pop, fitnesses, novelties, bcs, γ, num_elites)
        cache_elites!(sc, mi, elites)
        elite_seeds = [e[:seeds] for e in elites]
        @test elites[1][:seeds] in keys(sc)
        @test all(elite(p) in keys(sc) for p in pop)
        @test pop[argmax(fitnesses)] in pop
        @test length(findall(x->x==pop[argmax(fitnesses)], pop)) == 1
    end
end
