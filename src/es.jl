module ES

export compute_centered_ranks, NoiseTable, compute_grad, get_noise

using Flux
using StableRNGs

function compute_ranks(x)
  @assert ndims(x) == 1
  ranks = zeros(Int, size(x))
  ranks[sortperm(x)] = 1:length(x)
  ranks
end
function compute_centered_ranks(x)
  ranks = (compute_ranks(x) .- 1) / length(x)
  ranks = ranks .- 0.5
  ranks
end

mutable struct NoiseTable
  rng::StableRNGs.LehmerRNG
  noise::Vector{Float32}
  nparams::Int
  pop_size::Int
  σ::Float32
end

NoiseTable(rng::StableRNGs.LehmerRNG, nparams::Int, pop_size::Int, σ::Float32) = NoiseTable(rng, σ * randn(rng, Float32, nparams + pop_size), nparams, pop_size, σ)
get_noise(nt::NoiseTable, idx::Int) = nt.noise[idx:idx+nt.nparams-1]
function refresh_noise!(nt)
  nt.noise = randn(nt.rng, Float32, nt.nparams + nt.pop_size) * nt.σ
end

function compute_grad(nt::NoiseTable, centered_ranks::Vector{Float32})
  @assert nt.pop_size == length(centered_ranks)
  grad = zeros(Float32, nt.nparams)
  for i in 1:nt.pop_size
    grad += get_noise(nt, i) * centered_ranks[i]
  end
  grad
end

function test_nt()
  rng = StableRNG(123)
  nt = NoiseTable(rng, 2, 4, 0.1f0)
  println(get_noise(nt, 1))
  println(get_noise(nt, 2))
  nt = NoiseTable(rng, 2, 4, 0.1f0)
  println(get_noise(nt, 1))
  println(get_noise(nt, 2))
end


end
