using EvoTrade
using Test
using Flux


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
    @test all(length.(pop) .== 3)
    for g in 2:2
        bcs[2] = bcs[2] .+ 1 # make sure that the second seed is always novel
        pop, elites = create_next_pop(g, sc, pop, fitnesses, novelties, bcs, γ, 2)
        best_f_mr = elites[1][:seeds][end-1]
        best_n_mr = elites[2][:seeds][end-1]
        cache_elites!(sc, mi, elites)
        # offspring mutation rates are between 0.5 and 2x the parent
        @test all(length.(pop) .== 1+2g)
        for i in 1:3
            @test 0.5*best_f_mr<= pop[i][end-1] <= 2.0*best_f_mr
        end
        for i in 4:6
            @test 0.5*best_n_mr<= pop[i][end-1] <= 2.0*best_n_mr < g
        end
    end
end

@testset "test_multi_gen_γ={0.001,0.999}" begin
    # assure there are no crashes at weird γ values
    obs_size = (30,32,4, 1)
    n_actions = 4
    pop_size = 6
    m = make_model(:small,
        obs_size,
        n_actions,
        vbn=false,
        lstm=false)
    for γ in [0.001, 0.999]
        sc = SeedCache(maxsize=4)
        mi = ModelInfo(m)
        p1 = [1.]
        p2 = [2.]
        pop = [p1, p2, [0.0], [0.0], [0.0], [0.0]]
        fitnesses = [2., 1., 0., 0., 0., 0.]
        novelties = [1., 2., 0., 0., 0., 0.]
        bcs = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
        pop, elites = create_next_pop(1, sc, pop, fitnesses, novelties, bcs, γ, 2)
        cache_elites!(sc, mi, elites)
        for g in 2:5
            pop, elites = create_next_pop(g, sc, pop, fitnesses, novelties, bcs, γ, 2)
            cache_elites!(sc, mi, elites)
        end
    end
end

@testset "test_mutation" begin
    for i in rand(10)
        m = M(i)
        @test 0.5i <= m <= 2i
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
   for g in 2:5
       bcs[2] = bcs[2] .+ 1 # make sure that the second seed is always novel
       pop, elites = create_next_pop(g, sc, pop, fitnesses, novelties, bcs, γ, 2)
       f_mr = elites[1][:seeds][end-1]
       f_mr2 = elites[2][:seeds][end-1]
       cache_elites!(sc, mi, elites)
       # offspring mutation rates are between 0.5 and 2x the parent
       @test all(length.(pop) .== 1+2g)
       for i in 1:6
           @test (0.5*f_mr<= pop[i][end-1] <= 2.0*f_mr) ||
           (0.5*f_mr2<= pop[i][end-1] <= 2.0*f_mr2)
       end
   end
   sc = SeedCache(maxsize=4)
   γ = 1.0
   pop = [p1, p2, [0.0], [0.0], [0.0], [0.0]]
   pop, elites = create_next_pop(1, sc, pop, fitnesses, novelties, bcs, γ, 2)
   cache_elites!(sc, mi, elites)
   @test elites[1][:seeds] == p2
   @test elites[2][:seeds] == p1
   for g in 2:5
       bcs[2] = bcs[2] .+ 1 # make sure that the second seed is always novel
       pop, elites = create_next_pop(g, sc, pop, fitnesses, novelties, bcs, γ, 2)
       n_mr = elites[1][:seeds][end-1]
       n_mr2 = elites[2][:seeds][end-1]
       cache_elites!(sc, mi, elites)
       # offspring mutation rates are between 0.5 and 2x the parent
       @test all(length.(pop) .== 1+2g)
       for i in 1:6
           @test 0.5*n_mr<= pop[i][end-1] <= 2.0*n_mr ||
           0.5*n_mr2<= pop[i][end-1] <= 2.0*n_mr2
       end
   end
end
