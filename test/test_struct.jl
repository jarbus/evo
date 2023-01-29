dummy_geno = [1.0, 0.5, 3.0] |> v32
dummy_bc = [1.0, 0.5, 3.0] |> v32
root_dir = dirname(@__FILE__)  |> dirname |> String
@testset "struct_creation" begin
  @testset "Ind" begin
    ind = Ind("1", dummy_geno)
    @test ind.id == "1"
    @test ind.geno == dummy_geno
    ind = Ind("1")
    @test length(ind.geno) == 1
    ind = Ind()
    @test ind.id != ""
    @test length(ind.geno) == 1
  end
  @testset "Pop" begin
    pop = Pop("1", 10)
    @test pop.id == "1"
    @test pop.size == 10
    @test 10 == length(pop.inds)
    @test pop.archive == Set{BC}()
    @test pop.elites == []
    @test pop.avg_walks == []
    @test pop.mets == Dict{String, Vector{Float64}}()
    @test pop.info == Dict{Any, Any}()
  end
end
@testset "struct_compression" begin
  es = [Ind("1", s |> v32) for s in 
            [[1,0.1,2,0.1,3,0.1,4,0.1,5,0.1,6],
             [1,0.1,2,0.1,3,0.1,5,0.1,8], [4] ]]
  @testset "last_matching_index" begin
    @test 5 == EvoTrade.find_last_matching_idx(es[1], es[2])
    @test 0 == EvoTrade.find_last_matching_idx(es[1], es[3])
  end
  @testset "compute_elite_idxs" begin
    eidxs= EvoTrade.compute_elite_idxs!(es)
    @test es[1].elite_idxs == Set{Int}([0, 5, 11])
    @test es[2].elite_idxs == Set{Int}([0, 5, 9])
  end
  @testset "get_elite_idxs" begin
    pop = Pop("1", 3, [Ind([e.geno; [0.1f0,32f0]]) for e in es])
    pop.elites = es
    pop_idxs = EvoTrade.get_elite_idxs(pop)
    for i in 1:length(pop.inds)
      @test pop_idxs[i] == es[i].elite_idxs
    end
  end
  @testset "compute_prefixes" begin
    elites = [Ind(s|>v32) for s in [[1], [2,.01,3,.01,5,.01,7],
                        [2,.01,3,.01,5,.01,6], [], [9,.01,9,.01,9],
                        [9,.01, 9,.01, 9,.01, 9,.01, 9,.01, 9]]]
    prefixes = compute_prefixes(elites)
    @test Set([[2,.01,3,.01,5]|>v32, [9,.01,9,.01,9]|>v32]) ⊆ Set(values(prefixes)) 
    @test string(hash([2,.01,3,.01,5]|>v32)) in keys(prefixes)
  end
  @testset "compress/decompress_pop" begin
    elites = [Ind(s|>v32) for s in [
      [1,0.1,3,0.1,4,0.1,5,0.1,6],
      [1,0.1,3,0.1,4,0.1,5,0.1,8]]]
    prefixes = compute_prefixes(elites)
    genomes = [[1,0.1,3,0.1,4,0.1,5,0.1,6,0.1,7]|>v32,
           [1,0.1,3,0.1,4,0.1,5,0.1,8,0.1,13]|>v32]
    pop = Pop("1",2,[Ind(g) for g in genomes])
    pop.elites = elites
    prefixes = compute_prefixes(elites)
    rollout_pop = compress_pop(pop, prefixes)
    @test rollout_pop[1].elite_idxs |> length == 2
    @test rollout_pop isa Vector{RolloutInd}
    decomp_pop = decompress_group(rollout_pop, prefixes)
    @test decomp_pop[1].geno == genomes[1]
    @test decomp_pop[2].geno == genomes[2]
    @testset "mk_id_player_map" begin
      id_map, counts = mk_id_player_map(decomp_pop)
      @test id_map[decomp_pop[1].id*"_1"]==decomp_pop[1].id
      @test id_map[decomp_pop[1].id*"_1"]==decomp_pop[1].id
      dup_pop = [decomp_pop[1], decomp_pop[1]]
      id_map, counts = mk_id_player_map(dup_pop)
      @test id_map[dup_pop[1].id*"_1"] == dup_pop[1].id
      @test id_map[dup_pop[2].id*"_2"] == dup_pop[2].id
    end
  end
end

@testset "next_pop" begin
  pop = Pop("1", 4)
  γ=0.5
  n_elites = 2
  fits = [4, 3, 2, 1]
  novs = [1, 2, 3, 4]
  for (i, (f, n)) in enumerate(zip(fits, novs))
    pop.inds[i].fitnesses = [f]
    pop.inds[i].novelty = n
  end
  next_pop = create_next_pop(pop, γ, n_elites)
  @test next_pop.size == 4
  e_genos = [e.geno for e in next_pop.elites]
  for ind in next_pop.inds
    @test elite(ind.geno) ∈ e_genos
  end
  @test e_genos[1] == next_pop.inds[1].geno
  @test e_genos[2] == next_pop.inds[2].geno
  @test pop.inds[4].geno == e_genos[2]
  @test length(next_pop.inds[3].geno) == 3
  @test length(next_pop.inds[4].geno) == 3
  @test length(next_pop.inds[1].bcs) == 0
end

@testset "add_to_archive!" begin
  pop = Pop("1", 4)
  γ=0.5
  n_elites = 2
  fits = [4, 3, 2, 1]
  pop.inds[1].bcs = [[1]]
  add_to_archive!(pop, 1.00)
  @test [1] ∈ pop.archive
  pop.inds[2].bcs = [[2]]
  add_to_archive!(pop, 0.00)
  @test [2] ∉ pop.archive
end

@testset "novelties" begin
  pop = Pop("1", 4)
  for i in 1:4
    pop.inds[i].bcs = [rand(Float32, 7), rand(Float32, 7)]
  end
  push!(pop.archive, rand(Float32, 7))
  @testset "make_bc_matrix" begin
    bc_mat, bc_ids = EvoTrade.make_bc_matrix(pop)
    @test size(bc_mat) == (7, 9)
    @test bc_ids[1] == [1,2]
    @test bc_ids[4] == [7,8]
    @test bc_mat[:,1] == pop.inds[1].bcs[1]
    @test bc_mat[:,8] == pop.inds[4].bcs[2]
  end
  @testset "compute_novelties!" begin
    for ind in pop.inds
      @test ismissing(ind.novelty)
    end
    compute_novelties!(pop)
    for ind in pop.inds
      @test !ismissing(ind.novelty)
    end
  end
end
