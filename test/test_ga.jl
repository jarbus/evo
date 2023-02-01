using NearestNeighbors

function euclidist(v1::Vector, v2::Vector)::Float64
    return sqrt(sum((v1 .- v2).^2))
end

@testset "test_groups" begin
  pops = [Pop(string(i), 4) for i in 1:2]
  prefixes = Prefixes()
  compops = compress_pops(pops, prefixes)
  @testset "random_groups" begin
    groups = random_groups(compops..., 
              rollout_group_size=2,
              rollouts_per_ind=2)
    @test length(groups) == 8
    end
  @testset "test_all_v_best" begin
    groups = all_v_best(compops..., 
              rollout_group_size=2,
              rollouts_per_ind=2)
    @test length(groups) == 16
  end
end
