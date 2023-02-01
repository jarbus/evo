BC = Vector{Float32}
F = Float32
Novelty = Float32
Seed = UInt32
Geno = Vector{Float32}
CompGeno = Vector{Union{String, Float32}}
EliteIdxs = Set{Int}
V32 = Vector{Float32}
Walk = Vector{Tuple{Float32, Float32}}
Prefixes = Dict{String, Geno}

mutable struct Ind4
  id::String
  geno::Geno
  bcs::Vector{BC}
  bc::Optional{BC} # some combination of bcs
  novelty::Optional{F}
  fitnesses::Vector{F}
  fitness::Optional{F} # some combination of fitnesses
  elite_idxs::Optional{EliteIdxs}
  walks::Vector{Walk}
end
Ind4(id::String, geno::Geno, eidx::EliteIdxs) =
Ind4(id, geno, [], missing, missing, [], missing, eidx, Vector{Walk}())
Ind4(id::String, geno::Geno) =
  Ind4(id, geno, EliteIdxs())
Ind4(id::String) = Ind4(id, rand(Seed,1) |> v32)
# for making tests less painful
Ind4(geno::Geno) = Ind4(randstring(8), geno)
Ind4() = Ind4(randstring(8))
Ind4(ind::Ind4) = Ind4(ind.id, deepcopy(ind.geno), deepcopy(ind.elite_idxs))
Ind = Ind4

function mk_id_map(inds::Vector{Ind})
  id_map = Dict{String, Int}()
  for (i, ind) in enumerate(inds)
    id_map[ind.id] = i
  end
  id_map
end

mutable struct Pop6
  id::String
  size::Int
  inds::Vector{Ind}
  id_map::Dict{String, Int}
  archive::Set{BC}
  elites::Vector{Ind}
  avg_walks::Vector{Vector{Float64}}
  mets::Dict{String, Vector{Float64}}
  info::Dict{Any, Any}
end
function Pop6(id::String, size::Int, inds::Vector{Ind})
  Pop6(id, size, inds, mk_id_map(inds), Set{BC}(), [], [],
        Dict{String, Vector{Float64}}(), Dict{Any, Any}())
end
Pop6(id::String, size::Int) = 
  Pop6(id, size, [Ind("$id-$i") for i in 1:size])
Pop = Pop6

struct RolloutInd1
  id::String
  geno::CompGeno
  elite_idxs::EliteIdxs
end
RolloutInd = RolloutInd1

struct Batch3
  rews::Dict{<:Any, <:Any}
  mets::Dict{<:Any, <:Any}
  bcs::Dict{<:Any, <:Any}
  info::Dict{<:Any, <:Any}
end
Batch = Batch3
