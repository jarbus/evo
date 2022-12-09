module NoiseTables
export NoiseTable, compute_grad, get_noise, refresh_noise!, reconstruct
using StableRNGs
mutable struct NoiseTable
  rng::StableRNGs.LehmerRNG
  noise::Vector{Float32}
  nparams::Int
  pop_size::Int
  σ::Float32
end

NoiseTable(rng::StableRNGs.LehmerRNG, nparams::Int, pop_size::Int, σ::Float32) = NoiseTable(rng, σ * randn(rng, Float32, nparams + pop_size), nparams, pop_size, σ)
get_noise(nt::NoiseTable, idx::UInt) = @view nt.noise[idx:idx+nt.nparams-1]
function refresh_noise!(nt::NoiseTable)
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

function reconstruct(nt::NoiseTable, x::Vector{UInt}, ϵ::Float32=0.01f0)
  theta = zeros(Float32, nt.nparams)
  theta .+= get_noise(nt, x[1]) ./ 32f0
  for seed in x[2:end]
    theta .+= get_noise(nt, seed)
  end
  theta .* ϵ
end



end
