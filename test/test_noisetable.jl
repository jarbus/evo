using StableRNGs

function add!(x, y)
  @inbounds for i in 1:length(x)
    x[i] += y[i]
  end
end


@testset "add_noise!" begin
  model = Chain(Dense(1, 1), Dense(1,2))
  mi = ModelInfo(model)
  nt = NoiseTable(StableRNG(1), length(mi), 1f0)
  @test mi.starts_and_ends == [(1,2),(2,3),(3,5),(5,7)]
  core = MutCore(1, 1f0, Set([1,2,3,4]))
  old_params = EvoTrade.gen_params(StableRNG(1), mi, 1)
  # Check that all layers get updated
  params = copy(old_params)
  EvoTrade.add_noise!(nt, mi, params, core)
  @test all(params .!= old_params)
  # Check that only some layers get updated
  core = MutCore(1, 1f0, Set([1]))
  params = copy(old_params)
  EvoTrade.add_noise!(nt, mi, params, core)
  @test params[1] != old_params
  @test params[2:end] == old_params[2:end]
  # Check that a seed of 2 gives a different result
  core1 = MutCore(1, 1f0, Set([1,2,3,4]))
  core2 = MutCore(2, 1f0, Set([1,2,3,4]))
  params1 = copy(old_params)
  params2 = copy(old_params)
  EvoTrade.add_noise!(nt, mi, params1, core1)
  EvoTrade.add_noise!(nt, mi, params2, core2)
  @test params1 != params2
  # Check that a different mutation rate gives a different result
  core1 = MutCore(1, 1f0, Set([1,2,3,4]))
  core2 = MutCore(1, 2f0, Set([1,2,3,4]))
  params1 = copy(old_params)
  params2 = copy(old_params)
  EvoTrade.add_noise!(nt, mi, params1, core1)
  EvoTrade.add_noise!(nt, mi, params2, core2)
  @test params1 != params2
end
