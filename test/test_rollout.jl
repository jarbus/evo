root_dir = dirname(@__FILE__)  |> dirname |> String

@testset "mk_mod" begin
  ind1 = Ind("1", [1.0, 0.5, 3.0, 0.5, 4.0] |> v32)
  ind1.elite_idxs = Set{Int}([3, 5])
  ind2 = Ind("2", [1.0, 0.5, 3.0] |> v32)
  ind2.elite_idxs = Set{Int}([3])
  sc = SeedCache(maxsize=10)
  obs_size = (10, 10, 3, 1)
  n_actions = 9
  m = make_model(:small, obs_size, n_actions, lstm=false)
  θ, re = Flux.destructure(m)
  mi = ModelInfo(m, re)
  models, _, _ = mk_mods(sc, mi, [ind1, ind1])
  @test length(models) == 2
  @test collect(keys(models)) == ["1_1", "1_2"]
  @test Flux.params(models["1_1"]) == Flux.params(models["1_2"])
  models, _, _ = mk_mods(sc, mi, [ind1, ind2])
  @test length(models) == 2
  @test collect(keys(models)) == ["1_1", "2_1"]
  @test Flux.params(models["1_1"]) != Flux.params(models["2_1"])
end

@testset "test_rollout_trade" begin
    """For 1v1 case, we want to make sure that agents are playing both sides of the game"""
    expname = ["--exp-name", "test", "--cls-name","test", "--local", "--datime", "test"]
    arg_vector = read("$root_dir/afiles/cls-test/test-ga-trade.arg", String) |> split
    args = parse_args(vcat(arg_vector, expname), get_arg_table())
    env_config = mk_env_config(args)
    env = PyTrade().Trade(env_config)

    sc = SeedCache(maxsize=10)
    ind1 = Ind("1_1", dummy_geno, Set{Int}([3, 5]))
    m = make_model(:small,
            (env.obs_size..., 2),
            env.num_actions,
            lstm=true)
    θ, re = Flux.destructure(m)
    mi = ModelInfo(m, re)
    models, id_map, _ = mk_mods(sc, mi, [ind1, ind1])
    gamebatch = run_batch(env_config, models, args, batch_size=2)
    idbatch = process_batch(gamebatch, id_map, true)
    @test "1_1" in keys(idbatch.rews)
    @test length(idbatch.rews["1_1"]) == 2
    pop = Pop("1", 2, [ind1, ind1])
    metrics = aggregate_rollouts!([idbatch], [pop])
    @test length(metrics) > 0
end
