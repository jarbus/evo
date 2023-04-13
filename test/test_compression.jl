@testset "compute_compression_data" begin
  g0 = [Mut(UInt32(10), 1f0)]
  g1 = [Mut(UInt32(1), 1f0), Mut(UInt32(1), 1f0)]
  g2 = [Mut(UInt32(1), 1f0), Mut(UInt32(2), 1f0)]
  g3 = [Mut(UInt32(2), 1f0), Mut(UInt32(3), 1f0)]
  g4 = [Mut(UInt32(3), 1f0), Mut(UInt32(4), 1f0),
        Mut(UInt32(5), 1f0), Mut(UInt32(6), 1f0)]
  i0 = Ind("0", g0)
  i1 = Ind("1", g1)
  i2 = Ind("2", g2)
  i3 = Ind("3", g3)
  i4 = Ind("3", g4)
  pop = Pop("1", 5, [i0, i1, i2, i3, i4])
  pop.elites = pop.inds
  idxs = Evo.get_elite_idxs(pop)
  sorted_idxs = [sort(Int.(collect(i.elite_idxs))) for i in pop.inds]
  @test sorted_idxs[1] == [1]
  @test sorted_idxs[2] == [1, 2]
  @test sorted_idxs[3] == [1, 2]
  @test sorted_idxs[4] == [2]
  @test sorted_idxs[5] == [4]
  prefixes = Evo.compute_prefixes([pop])
  compop = compress_pop(pop, prefixes)
  dc = Evo.decompress_group(compop, prefixes)
  @test dc[1].geno == g0
  @test dc[2].geno == g1
  @test dc[3].geno == g2
  @test dc[4].geno == g3
  @test dc[5].geno == g4


  # ind = Ind("1", [1,2,3,4,5,6,7,8,9,10]|>v32)
  # pop = Pop("1", 1, [ind])
  # prefixes = Prefixes()
  # compop = compress_pop(pop, prefixes)
  # bytes = compute_compression_data(compop, prefixes)
  # @test 0 == bytes.compressed 
  # println(bytes)
  # prefix = ind.geno[1:7]
  # prefixes[string(hash(prefix))] = prefix
  # compop1 = compress_pop(pop, prefixes)
  # bytes1 = compute_compression_data(compop1, prefixes)
  # @test 0 != bytes1.compressed 
  # pop = Pop("2", 2, [ind, ind])
  # compop2 = compress_pop(pop, prefixes)
  # bytes2 = compute_compression_data(compop2, prefixes)
  # @test bytes2.compressed == 2 * bytes1.compressed
  #
  # bytes12 = compute_compression_data([compop1, compop2], prefixes)
  # @test bytes12.compressed == bytes1.compressed + bytes2.compressed
end
