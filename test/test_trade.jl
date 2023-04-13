using Evo
using Test
using PyCall
root_dir = dirname(@__FILE__)  |> dirname |> String

#@testset "test_trade" begin
#    expname = ["--exp-name", "test", "--cls-name","test", "--local", "--datime", "test"]
#    arg_vector = read("$root_dir/afiles/cls-test/test-ga-trade.arg", String) |> split
#    args = parse_args(vcat(arg_vector, expname), get_arg_table())
#    env_config = mk_env_config(args)
#    env = PyTrade().Trade(env_config)
#    @test env isa PyObject
#    Evo.Trade.reset!(env)
#end
#
#@testset "test_75_daystep" begin
#    expname = ["--exp-name", "test", "--cls-name","test", "--local", "--datime", "test"]
#    arg_vector = read("$root_dir/afiles/daystep-test/test-1atrade.arg", String) |> split
#    args = parse_args(vcat(arg_vector, expname), get_arg_table())
#    env_config = mk_env_config(args)
#    env = PyTrade().Trade(env_config)
#    @test env isa PyObject
#    Evo.Trade.reset!(env)
#    min_light = 0.0
#    max_light = 0.0
#    for i in 1:args["episode-length"]*3
#        pycall(env.light.step_light, Nothing)
#        ff = pycall(env.light.fire_frame, PyArray)
#        min_light = min(min_light, minimum(ff))
#        max_light = max(max_light, maximum(ff))
#    end
#    @test min_light == -1
#    @test max_light == 1
#end
#
#@testset "test_trade_seed" begin
#    expname = ["--exp-name", "test", "--cls-name","test", "--local", "--datime", "test"]
#    seed = ["--seed", "123"]
#    arg_vector = read("$root_dir/afiles/cls-test/test-ga-trade.arg", String) |> split
#
#    # no seed table
#    args = parse_args(vcat(arg_vector, expname), get_arg_table())
#    env_config = mk_env_config(args)
#    pt = PyTrade()
#    env = pt.Trade(env_config)
#    @test env isa PyObject
#    Evo.Trade.reset!(env)
#    no_seed_table = env.table
#
#
#    args = parse_args(vcat(arg_vector, expname, seed), get_arg_table())
#    env_config = mk_env_config(args)
#    @test "seed" in keys(env_config)
#
#    env1 = pt.Trade(env_config)
#    Evo.Trade.reset!(env1)
#    @test env1 isa PyObject
#    seed_table_1 = env1.table
#    
#
#    env2 = pt.Trade(env_config)
#    @test env2 isa PyObject
#    Evo.Trade.reset!(env2)
#    seed_table_2 = env2.table
#
#    @test seed_table_1 != no_seed_table
#    @test seed_table_1 == seed_table_2
#end

# THIS IS VALID, just commented out
#@testset "test_campfire_frame" begin
#    expname = ["--exp-name", "test", "--cls-name","test", "--local", "--datime", "test"]
#    arg_vector = read("$root_dir/afiles/daystep-test/test-1atrade.arg", String) |> split
#    args = parse_args(vcat(arg_vector, expname), get_arg_table())
#    env_config = mk_env_config(args)
#    env = PyTrade().Trade(env_config)
#    @test env isa PyObject
#    Evo.Trade.reset!(env)
#    cff = env.light.campfire_frame
#    @test maximum(cff) == 1.0
#    @test minimum(cff) == 0.0
#
#    for a in 1:1000
#        acts = [rand(1:9) for i in 1:args["episode-length"]]
#        env = PyTrade().Trade(env_config)
#        Evo.Trade.reset!(env)
#        for i in 1:10
#            obs, rew, done = Evo.Trade.step!(env, Dict("f0a0" => acts[i]))
#            Evo.Trade.render(env, "/dev/null")
#        end
#    end
#end


@testset "test_trade" begin
    expname = ["--exp-name", "test", "--cls-name","test", "--local", "--datime", "test"]
    arg_vector = read("$root_dir/afiles/daystep-test/test-1atrade.arg", String) |> split
    args = parse_args(vcat(arg_vector, expname), get_arg_table())
    env_config = mk_env_config(args)

    i1 = "1-4-grdg"
    i2 = "2-31-123fsdf"
    env_config["matchups"] = [(i1, i2)]
    env_config["grid"] = (1,1)
    env_config["foods"] = [(0,0,0), (1,0,0)]
    env_config["fires"] = [(0,0,0)]
    acts = [rand(1:9) for i in 1:args["episode-length"]]
    env = PyTrade().Trade(env_config)
    Evo.Trade.reset!(env)
    @test 1.0f0 == pycall(env.collection_modifier, Float32, i1, 0)
    @test 0.5f0 == pycall(env.collection_modifier, Float32, i1, 1) 
    @test 0.5f0 == pycall(env.collection_modifier, Float32, i2, 0)
    @test 1.0f0 == pycall(env.collection_modifier, Float32, i2, 1)
    # 0-3 is direction
    # 4,5,6,7 are pick0, place0, pick1, place1
    Evo.Trade.step!(env, Dict(i1 => 4))
    @test env.agent_food_counts[i1][1] == 4.9
    @test env.mc.picked_counts[i1][1] == 5
    @test env.agent_food_counts[i1][2] == 0.0
    @test env.mc.picked_counts[i1][2] == 0
    Evo.Trade.step!(env, Dict(i2 => 6))
    @test env.agent_food_counts[i2][1] == 0.0
    @test env.mc.picked_counts[i2][1] == 0
    @test env.agent_food_counts[i2][2] == 4.9
    @test env.mc.picked_counts[i2][2] == 5

    Evo.Trade.reset!(env)
    Evo.Trade.step!(env, Dict(i1 => 6))
    @test env.agent_food_counts[i1][1] == 0.0
    @test env.mc.picked_counts[i1][1] == 0
    @test env.agent_food_counts[i1][2] == 2.4
    @test env.mc.picked_counts[i1][2] == 2.5
    Evo.Trade.step!(env, Dict(i2 => 4))
    @test env.agent_food_counts[i2][1] == 2.4
    @test env.mc.picked_counts[i2][1] == 2.5
    @test env.agent_food_counts[i2][2] == 0.0
    @test env.mc.picked_counts[i2][2] == 0

    # for i in 1:10
    #     obs, rew, done = Evo.Trade.step!(env, Dict("f0a0" => acts[i]))
    #     Evo.Trade.render(env, "/dev/null")
    # end
end
