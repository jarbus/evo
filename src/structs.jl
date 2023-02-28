import Base: Dict, rand, ==, show, hash

BC = Vector{Float32}
F = Float32
Novelty = Float32
Seed = UInt32
MR = Float32
new_mr()::MR = 0.002#rand(Float32) * 0.0095f0 + 0.0005f0
EliteIdxs = Set{UInt32}
V32 = Vector{Float32}
Walk = Vector{Tuple{Float32, Float32}}


struct ModelInfo
  sizes::Vector{Tuple}
  biases::Vector{Bool}
  starts_and_ends::Vector{Tuple{UInt32,UInt32}}
  names::Vector{String}
  re::Flux.Optimisers.Restructure
end

struct MutCore
  seed::Seed
  mr::MR
  layers::Set{UInt32}
end
struct MutBinding
  start::Optional{UInt32}
  geno::Vector{MutCore}
end
MutBinding() = MutBinding(missing, [])
struct Mut
  core::MutCore
  score::Optional{Float32}
  crossed_over::Bool
  binding::MutBinding
end
function Base.:(==)(a::MutCore, b::MutCore)
  a.seed == b.seed &&
  a.mr == b.mr &&
  a.layers == b.layers
end
function Base.:(==)(a::MutBinding, b::MutBinding)
  a.geno == b.geno &&
  a.start === b.start
end
function Base.:(==)(a::Mut, b::Mut)
  # use === for values that might be missing
  a.core == b.core &&
  a.binding == b.binding &&
  a.score === b.score
end
Base.show(io::IO, m::UInt32) = print(io, "$(Int(m))")
Base.show(io::IO, m::Mut) = print(io, "Mut($(m.core), $(m.crossed_over))")
function Base.show(io::IO, mc::MutCore)
  print(io, "Core($(Int(mc.seed)), $(round(Float64(mc.mr), digits=2)) $(collect(mc.layers)))")
end
function Base.hash(mc::MutCore, h::UInt)
  hash(mc.seed) + hash(mc.mr) + h
end
function Base.hash(m::Mut, h::UInt)
  hash(m.core)
end

Mut(core::MutCore, score::Optional{Real}, binding::MutBinding) =
  Mut(core, score, false, binding)
Mut(seed::Seed, mr::MR) = Mut(MutCore(seed, mr, Set{UInt32}()))
Mut(mi::ModelInfo) = Mut(MutCore(rand(Seed), new_mr(),
                      Set{UInt32}(rand(1:length(mi.sizes), 1))))
Mut(c::MutCore) = Mut(c, missing, MutBinding(missing, []))
Mut(m::Mut, mr::MR) = Mut(MutCore(m.core.seed, mr, m.core.layers),
                          m.score,
                          m.binding)
mark_crossover(m::Mut) = Mut(m.core, m.score, true, m.binding)
mark_score(m::Mut, score::Float32) = Mut(m.core, score, m.crossed_over, m.binding)
rand(::Type{Mut}) = Mut(rand(Seed), rand(MR))
rand(::Type{Mut}, n::Int) = [rand(Mut) for _ in 1:n]


GenePool = Vector{Mut}
struct GenePoolStatistics
  num_copied_muts::UInt32
  copied_layers_ratios::Vector{Float32}
  copied_layers_mrs::Vector{Vector{Float32}}
end
Geno = Vector{Mut}
CompGeno = Vector{Union{String, Mut}}
Prefixes = Dict{String, Geno}
SeedCache = LRU{Geno,V32}

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

function Ind4(id::String, mi::ModelInfo)
  core = MutCore(rand(Seed), 1f0, Set{UInt32}(1:length(mi.sizes)))
  Ind4(id, [Mut(core)])
end
Ind4(id::String, geno::Geno, eidx::EliteIdxs) =
Ind4(id, geno, [], missing, missing, [], missing, eidx, Vector{Walk}())
Ind4(id::String, geno::Geno) =
  Ind4(id, geno, EliteIdxs())
# for making tests less painful
Ind4(geno::Geno) = Ind4(randstring(8), geno)
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
Pop6(id::String, size::Int, mi::ModelInfo) = 
  Pop6(id, size, [Ind("$id-$i", mi) for i in 1:size])
Pop = Pop6
genos(pop::Pop) = [ind.geno for ind in pop.inds]

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

mutable struct ReconDataCollector
  num_reconstructions::Int
  num_recursions::Int
  time_deltas::Vector{Float32}
end
ReconDataCollector() = ReconDataCollector(0, 0, Float64[])
function Dict(rdc::ReconDataCollector) 
  Dict{String, Any}(
    "num_reconstructions" => rdc.num_reconstructions,
    "num_recursions" => rdc.num_recursions,
    "time_deltas" => rdc.time_deltas,
  )
end
