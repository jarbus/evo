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

@testset "test_campfire_frame" begin
    expname = ["--exp-name", "test", "--cls-name","test", "--local", "--datime", "test"]
    arg_vector = read("$root_dir/afiles/daystep-test/test-1atrade.arg", String) |> split
    args = parse_args(vcat(arg_vector, expname), get_arg_table())
    env_config = mk_env_config(args)
    env = PyTrade().Trade(env_config)
    @test env isa PyObject
    Evo.Trade.reset!(env)
    cff = env.light.campfire_frame
    @test maximum(cff) == 1.0
    @test minimum(cff) == 0.0

    frame_types = split("food1 food2 selfpos selffood1 selffood2 otherpos otherfood1 otherfood2 light campfire xpos ypos")
    for a in 1:1000
        acts = [rand(1:9) for i in 1:args["episode-length"]]
        env = PyTrade().Trade(env_config)
        Evo.Trade.reset!(env)
        for i in 1:10
            obs, rew, done = Evo.Trade.step!(env, Dict("f0a0" => acts[i]))
            Evo.Trade.render(env, "/dev/null")
            # for frame in 1:size(obs["f0a0"], 3)
                # println(frame_types[frame])
                # obs["f0a0"][:,:,frame] |> display
            # end
        end
    end
end
