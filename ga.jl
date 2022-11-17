module GANS
using StableRNGs
using Flux
export reconstruct, compute_novelty, bc1

function reconstruct(x::Vector{<:UInt32}, len, ϵ=0.1)
  @assert length(x) > 0
  @assert len > 0
  theta = Flux.glorot_normal(StableRNG(x[1]), len)
  for seed in x
    theta .+= ϵ .* Flux.glorot_normal(StableRNG(seed), len)
  end

  @assert theta .|> isnan |> any |> !
  theta
end

function test_reconstruct()
  z = reconstruct([3, 4, 5], 4)
  println(z)
end

function compute_novelty(ind_bc::Vector{<:Float64}, archive_and_pop::Vector{Vector{Float64}})
    # Assumptions: Novelty against self is zero, ind_bc is in archive_and_pop
    sum(sum((ind_bc .- bc) .^ 2) for bc in archive_and_pop) / (length(archive_and_pop) - 1)
end

function test_compute_novelty()
    function gen_dist(len) 
        x = rand(len)
        x ./ sum(x)
    end
    archive = [gen_dist(9) for _ in 1:1000]
    pop = [gen_dist(9) for _ in 1:1000]
    archive_and_pop = vcat(archive, pop)
    for ind_bc in pop
        compute_novelty(ind_bc, archive_and_pop)
    end
    archive_and_pop = [[0.0, 1.0], [1.0, 0.0]]
    ind = [1.0, 0.0]
    nov = compute_novelty(ind, archive_and_pop)
    @assert nov isa Float64
    @assert nov == 2.0
end

function bc1(x::Vector{<:Int}, num_actions=9)
    # count number of each element in x
    counts = zeros(Int, num_actions)
    for i in x
        counts[i] += 1
    end
    counts ./ length(x)
end

function test_bc1()
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    bc = bc1(x, 9)
    @assert bc isa Vector{Float64}
    @assert length(bc) == 9
    @assert bc |> sum |> isapprox(1.0)
    @assert all(bc .== 1/9)

    x = [1,1,1,1,1,1,1,1,1]
    bc = bc1(x, 9)
    @assert bc isa Vector{Float64}
    @assert bc == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
end

end
