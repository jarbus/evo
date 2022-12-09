module ES

export compute_centered_ranks

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

end
