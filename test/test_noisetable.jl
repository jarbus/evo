using StableRNGs

function add!(x, y)
  @inbounds for i in 1:length(x)
    x[i] += y[i]
  end
end

@testset "noisetable" begin
  n_params = 4_000_000
  nt = NoiseTable(StableRNG(1), n_params, 0.5f0)
  noise1 = nt[1]
  noise2 = nt[2]
  @test noise1 != noise2
  x = zeros(Float32, n_params)
  @time for i in 1:1000
    EvoTrade.add_noise!(nt, x, i)
  end
end
