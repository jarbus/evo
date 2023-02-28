root_dir = dirname(@__FILE__)  |> dirname |> String
@testset "integration" begin
  n_elites = 2
  n_iter = 20
  γ = 1.0f0
  rollout_group_size = 2
  rollouts_per_ind = 2
  sc = SeedCache(maxsize=10)
  n_actions = 9

  arg_vector = read("$root_dir/afiles/cls-test/test-ga-trade.arg", String) |> split

  expname = ["--exp-name", "test-int", "--cls-name", "test-int", "--local", "--datime", "fsdf"] # get rid of .arg
  args = parse_args(vcat(arg_vector, expname), get_arg_table())
  args["exploration-rate"] = 1.0
  env_config = mk_env_config(args)
  env = PyTrade().Trade(env_config)

  m = make_model(:small,
          (env.obs_size..., 2),
          env.num_actions,
          lstm=true)
  θ, re = Flux.destructure(m)
  mi = ModelInfo(m)
  nt = NoiseTable(StableRNG(1), length(mi), 0.5f0)
  prefixes = Prefixes([])

  pops = [Pop(string(i), 4, mi) for i in 1:2]
  for pop in pops, ind in pop.inds
    @test ind.geno[1].core.layers == Set{UInt32}(1:length(mi.sizes))
  end
  for iter in 1:n_iter
    println("iter $iter")
    compops = compress_pops(pops, prefixes)
    for (i, cp) in enumerate(compops)
      inds = decompress_group(cp, prefixes)
      for j in 1:length(inds)
        @test inds[j].geno == pops[i].inds[j].geno
      end
    end
    groups = all_v_best(compops..., 
                rollouts_per_ind=rollouts_per_ind,
                rollout_group_size=rollout_group_size)

    id_batches = Vector{Batch}()
    for group in groups
      dc = decompress_group(group, prefixes)

      models, id_map, rdc_dict = mk_mods(sc, mi, nt, dc)
      gamebatch = run_batch(env_config, models, args, batch_size=1)
      id_batch = process_batch(gamebatch, id_map, true)
      push!(id_batches, id_batch)
    end
    EvoTrade.update_pops!(pops, id_batches, γ)
    for pop in pops, ind in pop.inds
      @test length(ind.fitnesses) > 0
      @test length(ind.bcs) > 0
    end
    @test pops[1].inds[1].novelty > 0
    log_improvements(pops[1])
    EvoTrade.plot_bcs("bcs", pops, 3)
    next_pops = create_next_pop(mi, pops, γ, n_elites)


    for pop in next_pops, ind in pop.inds
      length(ind.geno) <= 1 && continue
      @test ind.geno[end].binding.start >= 1
      @test length(ind.geno[end].core.layers) == length(mi.sizes)
    end
    @test next_pops[1].elites[1].geno == next_pops[1].inds[1].geno
    pops = next_pops
  end
end
