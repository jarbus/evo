root_dir = dirname(@__FILE__)  |> dirname |> String
@testset "integration" begin
  pops = [Pop(string(i), 4) for i in 1:2]
  n_elites = 2
  γ = 0.5
  sc = SeedCache(maxsize=10)
  obs_size = (10, 10, 3, 1)
  n_actions = 9
  m = make_model(:small, obs_size, n_actions, lstm=false)
  θ, re = Flux.destructure(m)
  mi = ModelInfo(m, re)
  prefixes = compute_prefixes([])
  compops = compress_pops(pops, prefixes)
  groups = all_v_all(compops...)

  expname = ["--exp-name", "test", "--cls-name","test", "--local", "--datime", "test"]
  arg_vector = read("$root_dir/afiles/cls-test/test-ga-trade.arg", String) |> split
  args = parse_args(vcat(arg_vector, expname), get_arg_table())
  env_config = mk_env_config(args)
  env = PyTrade().Trade(env_config)

  sc = SeedCache(maxsize=10)
  m = make_model(:small,
          (env.obs_size..., 2),
          env.num_actions,
          lstm=true)
  θ, re = Flux.destructure(m)
  mi = ModelInfo(m, re)
  id_batches = Vector{Batch}()
  for group in groups
    dc = decompress_group(group, prefixes)
    models, id_map = mk_mods(sc, mi, dc)
    gamebatch = run_batch(env_config, models, args, batch_size=2)
    id_batch = process_batch(gamebatch, id_map, true)
    push!(id_batches, id_batch)
  end
  EvoTrade.update_pops!(pops, id_batches)
  for pop in pops, ind in pop.inds
    @test length(ind.fitnesses) > 0
    @test length(ind.bcs) > 0
  end
  @test pops[1].inds[1].novelty > 0
  next_pop = create_next_pop(pops[1], γ, n_elites)
  @test next_pop.elites[1].geno == next_pop.inds[1].geno
end
