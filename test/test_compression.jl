@testset "compute_compression_data" begin
  ind = Ind("1", [1,2,3,4,5,6,7,8,9,10]|>v32)
  pop = Pop("1", 1, [ind])
  prefixes = Prefixes()
  compop = compress_pop(pop, prefixes)
  bytes = compute_compression_data(compop, prefixes)
  @test 0 == bytes.compressed 
  println(bytes)
  prefix = ind.geno[1:7]
  prefixes[string(hash(prefix))] = prefix
  compop1 = compress_pop(pop, prefixes)
  bytes1 = compute_compression_data(compop1, prefixes)
  @test 0 != bytes1.compressed 
  pop = Pop("2", 2, [ind, ind])
  compop2 = compress_pop(pop, prefixes)
  bytes2 = compute_compression_data(compop2, prefixes)
  @test bytes2.compressed == 2 * bytes1.compressed

  bytes12 = compute_compression_data([compop1, compop2], prefixes)
  @test bytes12.compressed == bytes1.compressed + bytes2.compressed
end
