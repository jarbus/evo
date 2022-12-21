using EvoTrade
using Test
using NearestNeighbors

@testset "test_average_bc" begin
  @test EvoTrade.average_bc([[1,2,3], [1,2,3]]) == [1,2,3]
  @test EvoTrade.average_bc([[1,2,3], [3,4,5]]) == [2,3,4]
end

@testset "test_compute_novelty_trade" begin
    function gen_dist(len...) 
        x = rand(len...)
        x ./ sum(x)
    end
    archive = gen_dist(9,1000)
    pop = gen_dist(9,1000)
    archive_and_pop = hcat(archive, pop)
    for ind_bc in eachcol(pop)
      compute_novelty(Vector(ind_bc), archive_and_pop, k=25)
    end
    archive_and_pop = [0.0 1.0 ; 1.0 0.0]
    ind = [1.0, 0.0]
    nov = compute_novelty(ind, archive_and_pop, k=1)
    @test nov isa Float64
    @test nov == sqrt(2.0)
end

@testset "test_compute_novelty_maze" begin
  k = 2
  # make a function that computes euclidean distance
  function euclidist(v1::Vector, v2::Vector)::Float64
      return sqrt(sum((v1 .- v2).^2))
  end

  archive = Float32.([1 0; 3 0; 4 0]')
  pop = Float32.([0 4 ;1 4])
  archive_and_pop = hcat(archive, pop)
  kdtree = KDTree(archive_and_pop, leafsize=100000)

  nov1 = compute_novelty(pop[:,1], archive_and_pop, k=k)
  nov2 = compute_novelty(pop[:,2], archive_and_pop, k=k)
  @test nov1 - ((euclidist([0,1], [1,0]) + euclidist([0, 1],[3, 0]))/ k) < 0.01
  @test nov2 - ((euclidist([4,4], [4,0]) + euclidist([4, 4],[3, 0]))/ k) < 0.01
end


@testset "test_compute_novelties_maze" begin
  k = 2
  # make a function that computes euclidean distance
  function euclidist(v1::Vector, v2::Vector)::Float64
      return sqrt(sum((v1 .- v2).^2))
  end

  archive = Float32.([1 0; 3 0; 4 0]')
  pop = Float32.([0 4 ;1 4])
  archive_and_pop = hcat(archive, pop)

  nov1, nov2 = compute_novelties(pop, archive_and_pop, k=k)
  @test nov1 - ((euclidist([0,1], [1,0]) + euclidist([0, 1],[3, 0]))/ k) < 0.01
  @test nov2 - ((euclidist([4,4], [4,0]) + euclidist([4, 4],[3, 0]))/ k) < 0.01
end


@testset "test_bc1" begin
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    bc = bc1(x, 9)
    @test bc isa Vector{Float64}
    @test length(bc) == 9
    @test bc |> sum |> isapprox(1.0)
    @test all(bc .== 1/9)

    x = [1,1,1,1,1,1,1,1,1]
    bc = bc1(x, 9)
    @test bc isa Vector{Float64}
    @test bc == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
end

@testset "test_bc2" begin
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    bc = bc2(x, 9)
    @test bc isa Vector{Float64}
    @test length(bc) == 9
    @test bc |> sum |> isapprox(1.0)
    @test all(bc .== 1/9)
    
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2]
    bc = bc2(x, 9)
    @test length(bc) == 18
    @test isapprox((sum(bc)), 2.0)
    @test all(bc[1:9] .== 1/9)
    @test all(bc[10:18] .== [1/2, 1/2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    x = [1,1,1,1,1,1,1,1,1]
    bc = bc2(x, 9)
    @test bc == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    x2 = vcat(x, x, x, x, x, x, x, x, x)
    bc = bc2(x2, 9)
    @test length(bc) == 81
    for i in 1:9:81
        @test bc[i:i+8] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    end
end

@testset "test_bc_novelty" begin
    function euclidist(v1::Vector, v2::Vector)::Float64
        return sqrt(sum((v1 .- v2).^2))
    end
    bca = bc1([1, 1, 1, 2], 3)
    bcb = bc1([1, 2, 2, 2], 3)
    bcc = bc1([3, 3, 3, 3], 3)
    nov_a = compute_novelty(bca, hcat(bca, bcb), k=1)
    nov_b = compute_novelty(bcb, hcat(bca, bcb), k=1)
    nov_c = compute_novelty(bcc, hcat(bca, bcb, bcc), k=2)
    @test nov_a == nov_b == sqrt(0.5)
    @test nov_c > nov_a
end

@testset "create_next_pop_single_met" begin
  pop = [UInt32.([1, 2, 3, 4]), UInt32.([5, 6, 7, 8])]
  for i in 1:1000
    push!(pop, UInt32.([0,0,0,0]))
  end
  next_pop = create_next_pop(1, pop, 1)
  @test length(next_pop) == length(pop)
  @test [1, 2, 3, 4] in next_pop
  @test [1, 2, 3, 4] in [x[1:4] for x in next_pop]
  @test [5,6,7,8] ∉ [x[1:4] for x in next_pop]
  pop = [UInt32.([1, 2]), UInt32.([3, 4]), UInt32.([5, 6]), UInt32.([7, 8])]
  for i in 1:1000
    push!(pop, UInt32.([0,0]))
  end
  next_pop = create_next_pop(2, pop, 2)
  @test UInt32.([1, 2]) in next_pop
  @test any(pop[1] == np[1:2] for np in next_pop)
  @test any(pop[2] == np[1:2] for np in next_pop)
end
@testset "create_next_pop_adaptive" begin
  pop = [UInt32.([1, 2, 3, 4]), UInt32.([5, 6, 7, 8])]
  for i in 1:1000
    push!(pop, UInt32.([0,0,0,0]))
  end
  fitnesses = zeros(Float32, length(pop))
  novelties = zeros(length(pop))
  fitnesses[1] = 1.0f0
  novelties[2] = 1.0f0
  γ=0.5
  next_pop, _ = create_next_pop(1, pop, fitnesses, novelties, γ, 2)
  @test length(next_pop) == length(pop)
  @test [1, 2, 3, 4] ∉ next_pop
  @test [1, 2, 3, 4] in [x[1:4] for x in next_pop]
  @test [5,6,7,8] ∈ [x[1:4] for x in next_pop]
  @test [0,0,0,0] ∉ [x[1:4] for x in next_pop]
  γ=0.0
  next_pop, _ = create_next_pop(1, pop, fitnesses, novelties, γ, 1)
  @test length(next_pop) == length(pop)
  @test [1, 2, 3, 4] ∉ next_pop
  @test all([[1, 2, 3, 4] == x[1:4] for x in next_pop])
  @test [5,6,7,8] ∉ [x[1:4] for x in next_pop]
  @test [0,0,0,0] ∉ [x[1:4] for x in next_pop]


  γ=1.0
  next_pop, _ = create_next_pop(1, pop, fitnesses, novelties, γ, 1)
  @test length(next_pop) == length(pop)
  @test [5,6,7,8] ∉ next_pop
  @test all([[5,6,7,8] == x[1:4] for x in next_pop])
  @test [1,2,3,4] ∉ [x[1:4] for x in next_pop]
  @test [0,0,0,0] ∉ [x[1:4] for x in next_pop]

  γ=0.999
  next_pop, _ = create_next_pop(1, pop, fitnesses, novelties, γ, 2)
  γ=0.001
  next_pop, _ = create_next_pop(1, pop, fitnesses, novelties, γ, 2)
end
@testset "add_to_archive" begin
  archive = Set()
  BC = [0.0 for _ in 1:10000]
  pop = [0.0 for _ in 1:10000]
  @test length(archive) == 0
  @test length(BC) > 0
  @test length(pop) > 0
  add_to_archive!(archive, BC, pop, 0.01)
  @test length(archive) > 0
end

@testset "reorder!" begin
  novelties = [1, 0, 1, 2]
  F         = [1, 1, 4, 3]
  BC        = [1, 2, 3, 4]
  pop       = [1, 2, 3, 4]
  reorder!(novelties, F, BC, pop)
  @test novelties[1] >= novelties[2] >= novelties[3] >= novelties[4]
  @test BC[1] == 4 == pop[1]
  @test BC[2] == 1 == pop[2]
  @test BC[3] == 3 == pop[3]
  @test BC[4] == 2 == pop[4]
  @test F[1] == 3
  @test F[2] == 1
  @test F[3] == 4
  @test F[4] == 1
end

@testset "compute_elite" begin
  f(a,b) = a,b
  elite = compute_elite(f, collect(1:4), [1f0, 2f0, 3f0, 4f0], k=2, n=2)
  @test elite[1] == 4
  @test elite[2] == 4
end
