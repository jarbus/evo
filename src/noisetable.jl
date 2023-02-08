using StableRNGs
import Base: getindex
const NT_LENGTH = 125_000_000 # 1 GB 
struct NoiseTable
  rng::StableRNGs.LehmerRNG
  noise::Vector{Float32}
  nparams::Int
  σ::Float32
end

function NoiseTable(rng::StableRNGs.LehmerRNG, nparams::Int, σ::Float32) 
  NoiseTable(rng, σ .* randn(rng, Float32, NT_LENGTH), nparams, σ)
end
function getindex(nt::NoiseTable, idx::Int) 
  @assert idx > 0
  nt[UInt32(idx)]
end
function getindex(nt::NoiseTable, idx::UInt32) 
  start_idx = (idx % (NT_LENGTH - nt.nparams)) + 1
  end_idx   = start_idx + nt.nparams
  @view nt.noise[start_idx:end_idx]
end
function add_noise!(nt::NoiseTable, params::Vector{Float32}, idx::UInt32) 
  noise = nt[idx]
  @inbounds @simd for i in 1:length(params)
    params[i] += noise[i]
  end
end
add_noise!(nt::NoiseTable, params::Vector{Float32}, idx::Int) = add_noise!(nt, params, UInt32(idx))
