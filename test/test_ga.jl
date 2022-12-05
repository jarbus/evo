@testset "test_reconstruct" begin
  z = reconstruct(UInt32.([3, 4, 5]), 4)
  @test length(z) == 4
end

@testset "test_compute_novelty_trade" begin
    function gen_dist(len) 
        x = rand(len)
        x ./ sum(x)
    end
    archive = [gen_dist(9) for _ in 1:1000]
    pop = [gen_dist(9) for _ in 1:1000]
    archive_and_pop = vcat(archive, pop)
    for ind_bc in pop
        compute_novelty(ind_bc, archive_and_pop, k=25)
    end
    archive_and_pop = [[0.0, 1.0], [1.0, 0.0]]
    ind = [1.0, 0.0]
    nov = compute_novelty(ind, archive_and_pop, k=1)
    @test nov isa Float64
    @test nov == 2.0
end

@testset "test_compute_novelty_maze" begin
  k = 2
  archive = [(1,0), (3,0), (4,0)]
  pop = [(0,0), (1,0)]
  archive_and_pop = vcat(archive, pop)
  @test compute_novelty(pop[1], archive_and_pop, k=k) == (3^2 + 4^2) / k
  @test compute_novelty(pop[2], archive_and_pop, k=k) == (2^2 + 3^2) / k
  k = 25
  archive = [(1,0) for i in 1:30]
  pop = [(0,0) for i in 1:30]
  archive_and_pop = vcat(archive, pop)
  @test compute_novelty(pop[1], archive_and_pop, k=k) == 1.0
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

@testset "test_bc_novelty" begin
    bca = bc1([1, 1, 1, 2], 3)
    bcb = bc1([1, 2, 2, 2], 3)
    bcc = bc1([3, 3, 3, 3], 3)
    nov_a = compute_novelty(bca, [bca, bcb], k=1)
    nov_b = compute_novelty(bcb, [bca, bcb], k=1)
    nov_c = compute_novelty(bcc, [bca, bcb, bcc], k=2)
    @test nov_a == nov_b == 0.5
    @test nov_c > nov_a
end

@testset "create_next_pop" begin
  pop = [UInt32.([1, 2, 3, 4]), UInt32.([5, 6, 7, 8])]
  next_pop = create_next_pop(1, pop, 1)
  @test length(next_pop) == 2
  @test [1, 2, 3, 4] in next_pop
end

@testset "add_to_archive" begin
  archive = Set()
  BC = [0.0 for _ in 1:10000]
  pop = [0.0 for _ in 1:10000]
  @test length(archive) == 0
  @test length(BC) > 0
  @test length(pop) > 0
  add_to_archive!(archive, BC, pop)
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
