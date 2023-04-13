root_dir = dirname(@__FILE__)  |> dirname |> String
function make_random_idbatch(id::String)
  Batch(Dict(id => rand(Float32, 1)),
       Dict(),
       Dict(id => [rand(Float32,1)]),
       Dict("avg_walks"=>Dict(id=>[[]]))
       )
end
@testset "test_fast_compression" begin
  ancestor = rand(1f0:4f0, 2_001)
  pops = [Pop(string(1), 20_000)]
  for pop in pops, ind in pop.inds
    ind.geno = deepcopy(ancestor)
  end
  n_elites = 2
  n_iter = 3
  γ = 1.0
  rollout_group_size = 2
  rollouts_per_ind = 2
  sc = SeedCache(maxsize=10)
  n_actions = 9

  arg_vector = read("$root_dir/afiles/maze-test/full-adaptive.arg", String) |> split

  expname = ["--exp-name", "fix_throttle.jl", "--cls-name", "fix_throttle", "--local", "--datime", "fsdf"] # get rid of .arg
  args = parse_args(vcat(arg_vector, expname), get_arg_table())
  args["exploration-rate"] = 1.0
  env_config = mk_env_config(args)
  env = maze_from_file(args["maze"])

  m = make_model(:small,
          (env.obs_size..., 2),
          env.num_actions,
          lstm=true)
  θ, re = Flux.destructure(m)
  mi = ModelInfo(m, re)
  nt = NoiseTable(StableRNG(1), length(mi), 0.5f0)

  for iter in 1:n_iter
    prefixes = compute_prefixes(pops)
    compops = compress_pops(pops, prefixes)
    for (i, cpop) in enumerate(compops), (j, rind) in enumerate(cpop)
      ind = decompress_group([rind], prefixes)[1]
      @test ind.geno == pops[i].inds[j].geno
    end
    id_batches = [make_random_idbatch(ind.id) for pop in pops for ind in pop.inds]
    Evo.update_pops!(pops, id_batches)
    #Evo.plot_bcs("bcs", pops, 3)
    next_pops = create_next_pop(pops, γ, n_elites)
    pops = next_pops
  end
end
