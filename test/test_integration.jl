@testset "test_integration" begin
  pop_size = 100
  n_elites = 3
  model_size = 10_000
  param_cache::SeedCache = SeedCache(maxsize=n_elites*3)
  m = make_model(:large, (11, 11, 7, 10), 4, lstm=true)
  mi = ModelInfo(m)
  pop = [rand(1.0f0:1000.0f0, 1) for _ in 1:pop_size]
  elites = []
  fitnesses = rand(pop_size)
  novelties = rand(pop_size)
  bcs = [rand(3) for _ in 1:pop_size]
  γ=0.5

  pop, elites = create_next_pop(1, param_cache, pop, fitnesses, novelties, bcs, γ, n_elites)
  cache_elites!(param_cache, mi, elites)

  function get_groups(pop, elites)
    prefixes = compute_prefixes(elites)
    rollout_pop = add_elite_idxs(pop, elites)
    compop = compress_pop(rollout_pop, prefixes)
    rollout_elites = add_elite_idxs([e[:seeds] for e in elites], elites)
    compelites = compress_pop(rollout_elites, prefixes)
    groups = create_rollout_groups(compop, compelites, 10, 10)
    groups, prefixes
  end

  ef, efp, en, enp = nothing, nothing, nothing, nothing
  for i in 2:3
      pop, elites = create_next_pop(i, param_cache, pop, fitnesses, novelties, bcs, γ, n_elites)
      if i > 2
        eseeds = [e[:seeds] for e in elites]
        @test ef in eseeds
        @test reconstruct(param_cache, mi, ef) == efp
        @test en in eseeds
        @test reconstruct(param_cache, mi, en) == enp
      end

      fitnesses = rand(pop_size)
      novelties = rand(pop_size)
      bcs = [rand(3) for _ in 1:pop_size]
      ef = pop[argmax(fitnesses)]
      en = pop[argmax(novelties)]
      efp = reconstruct(param_cache, mi, ef)
      enp = reconstruct(param_cache, mi, en)

      cache_elites!(param_cache, mi, elites)
      groups, prefixes = get_groups(pop, elites)
      for group in groups
        d_group = decompress_group(group, prefixes)
        for ind in d_group
          id, seeds, e_idx = ind
          id > 0 && @test pop[id] == seeds
          id < 0 && @test elites[-id][:seeds] == seeds
          seeds == ef && @test reconstruct(param_cache, mi, seeds, e_idx) == efp
          seeds == en && @test reconstruct(param_cache, mi, seeds, e_idx) == enp
        end
      end
  end
end
