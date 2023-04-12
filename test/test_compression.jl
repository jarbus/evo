
#@testset "test_group_compression" begin
#  elites = [Dict(:seeds=>s) for s in [[1], [2,3], [2,4], [2,3,5,7], [2,3,5,6], [2,3,5,8]]]
#  groups = [[(1, e[:seeds], Set()) for e in elites] for i in 1:2]
#
#  # Test with no prefixes for gen 1
#  prefixes = Dict()
#  compressed_groups = compress_groups(groups, prefixes)
#  @test compressed_groups == groups
#  decompressed_groups = [decompress_group(cg, prefixes) for cg in compressed_groups]
#  @test decompressed_groups == groups
#
#  prefixes = compute_prefixes(elites)
#  compressed_groups = compress_groups(groups, prefixes)
#  @test length(compressed_groups) == 2
#  @test compressed_groups[1] == compressed_groups[2]
#  @test compressed_groups[1][1] == (1, [1], Set())
#  @test compressed_groups[1][2] == (1, [string(hash([2,3]))], Set())
#  @test compressed_groups[1][3] == (1, [string(hash([2])),4], Set())
#  @test compressed_groups[1][4] == (1, [string(hash([2,3,5])),7], Set())
#  @test compressed_groups[1][5] == (1, [string(hash([2,3,5])),6], Set())
#  @test compressed_groups[1][6] == (1, [string(hash([2,3,5])),8], Set())
#
#  decompressed_groups = [decompress_group(cg, prefixes) for cg in compressed_groups]
#  @test decompressed_groups == groups
#end

@testset "test_optimization_units" begin
  @testset "test_compute_elite_idxs" begin
    elites=[Dict(:seeds=>s) for s in [
      [1],
      [2,0.1,3],
      [2,0.1,4],
      [2,0.1,3,0.1,5,0.1,7],
      [2,0.1,3,0.1,5,0.1,6],
      [2,0.1,3,0.1,5,0.1,8],
      [],
      [9,0.1,9,0.1,9],
      [9,0.1,9,0.1,9,0.1,9,0.1,9,0.1,9]]]
    eidxs = Evo.GANS.compute_elite_idxs(elites)
    @test eidxs[elites[1][:seeds]] == Set{Int}([0,1])
    @test eidxs[elites[2][:seeds]] == Set{Int}([0,3,1])
    @test eidxs[elites[3][:seeds]] == Set{Int}([0,3,1])
    @test eidxs[elites[4][:seeds]] == Set{Int}([0,1,3,5,7])
    @test eidxs[elites[5][:seeds]] == Set{Int}([0,1,3,5,7])
    @test eidxs[elites[7][:seeds]] == Set{Int}([0])
    @test eidxs[elites[8][:seeds]] == Set{Int}([0,5])
    @test eidxs[elites[9][:seeds]] == Set{Int}([0,5,11])
  end
  @testset "test_compute_prefixes" begin
    elites = [Dict(:seeds=>s) for s in [[1], [2,.01,3,.01,5,.01,7], [2,.01,3,.01,5,.01,6], [], [9,.01,9,.01,9], [9,.01, 9,.01, 9,.01, 9,.01, 9,.01, 9]]]
    prefixes = compute_prefixes(elites)
    @test Set([[2,.01,3,.01,5], [9,.01,9,.01,9]]) ⊆ Set(values(prefixes)) 
    @test string(hash([2,.01,3,.01,5])) in keys(prefixes)
  end
  @testset "test_add_elite_idxs" begin
    elites = [Dict(:seeds=>s) for s in [[1,0.1,2,0.1,3,0.1,4,0.1,5,0.1,6],[1,0.1,2,0.1,3,0.1,5,0.1,8], []]]
    pop = [vcat(elites[1][:seeds], [0.1,7]), vcat(elites[2][:seeds], [0.1,13])]
    epop = add_elite_idxs(pop, elites)
    @test epop[1][2] == Set{Int}([0, 5, 11])
    @test epop[2][2] == Set{Int}([0, 5, 9])
  end
  @testset "test_compression" begin
    elites = [Dict(:seeds=>s) for s in [
                            [1,0.1,3,0.1,4,0.1,5,0.1,6].|>Float32,
                            [1,0.1,3,0.1,4,0.1,5,0.1,8].|>Float32]]
    prefixes = compute_prefixes(elites)
    pop = [[1,0.1,3,0.1,4,0.1,5,0.1,6,0.1,7].|>Float32,
           [1,0.1,3,0.1,4,0.1,5,0.1,8,0.1,13].|>Float32]
    rollout_pop = add_elite_idxs(pop, elites)
    compop = compress_pop(rollout_pop, prefixes)
    @testset "test_compress_pop" begin
      @test compop[1][1] == vcat([string(hash(pop[1][1:end-4]))], pop[1][end-3:end])
      @test compop[2][1] == vcat([string(hash(pop[2][1:end-4]))], pop[2][end-3:end])
    end
    groups = create_rollout_groups(compop, 1, 1)
    @testset "test_create_rollout_groups" begin
    end
    @testset "test_decompress_group" begin
      group = decompress_group(groups[1], prefixes)
      @test group[1][2] == pop[1] || group[1][2] == pop[2]
      group = decompress_group(groups[2], prefixes)
      println(group[1][2])
      @test group[1][2] == pop[1] || group[1][2] == pop[2]
    end
  end
end

