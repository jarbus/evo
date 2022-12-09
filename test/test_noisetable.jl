using EvoTrade
using Test
using StableRNGs
@testset "test_NoiseTable" begin
  rng = StableRNG(123)
  nt = NoiseTable(rng, 2, 4, 0.1f0)
  @test get_noise(nt, 1) == get_noise(nt, 1)
  @test get_noise(nt, 1) != get_noise(nt, 2)

  old_noise = get_noise(nt, 1)
  refresh_noise!(nt)
  @test get_noise(nt, 1) != old_noise 
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
