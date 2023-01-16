@testset "test_compute_prefixes" begin
  elites = [Dict(:seeds=>s) for s in [[1], [2,3], [2,4], [2,3,5,7], [2,3,5,6], [2,3,5,8]]]
  prefixes = compute_prefixes(elites)
  @test Set(values(prefixes)) == Set([[2,3,5], [2,3], [2]])
end

@testset "test_group_compression" begin
  elites = [Dict(:seeds=>s) for s in [[1], [2,3], [2,4], [2,3,5,7], [2,3,5,6], [2,3,5,8]]]
  groups = [[(1, e[:seeds], Set()) for e in elites] for i in 1:2]

  # Test with no prefixes for gen 1
  prefixes = Dict()
  compressed_groups = compress_groups(groups, prefixes)
  @test compressed_groups == groups
  decompressed_groups = [decompress_group(cg, prefixes) for cg in compressed_groups]
  @test decompressed_groups == groups

  prefixes = compute_prefixes(elites)
  compressed_groups = compress_groups(groups, prefixes)
  @test length(compressed_groups) == 2
  @test compressed_groups[1] == compressed_groups[2]
  @test compressed_groups[1][1] == (1, [1], Set())
  @test compressed_groups[1][2] == (1, [string(hash([2,3]))], Set())
  @test compressed_groups[1][3] == (1, [string(hash([2])),4], Set())
  @test compressed_groups[1][4] == (1, [string(hash([2,3,5])),7], Set())
  @test compressed_groups[1][5] == (1, [string(hash([2,3,5])),6], Set())
  @test compressed_groups[1][6] == (1, [string(hash([2,3,5])),8], Set())

  decompressed_groups = [decompress_group(cg, prefixes) for cg in compressed_groups]
  @test decompressed_groups == groups

end
