using EvoTrade
using Test
using PyCall
root_dir = dirname(@__FILE__)  |> dirname |> String

@testset "test_trade" begin
    expname = ["--exp-name", "test", "--cls-name","test", "--local", "--datime", "test"]
    arg_vector = read("$root_dir/afiles/cls-test/test-ga-trade.arg", String) |> split
    args = parse_args(vcat(arg_vector, expname), get_arg_table())
    env_config = mk_env_config(args)
    env = PyTrade().Trade(env_config)
    @test env isa PyObject
    EvoTrade.Trade.reset!(env)
end

@testset "test_75_daystep" begin
    expname = ["--exp-name", "test", "--cls-name","test", "--local", "--datime", "test"]
    arg_vector = read("$root_dir/afiles/daystep-test/test-1atrade.arg", String) |> split
    args = parse_args(vcat(arg_vector, expname), get_arg_table())
    env_config = mk_env_config(args)
    env = PyTrade().Trade(env_config)
    @test env isa PyObject
    EvoTrade.Trade.reset!(env)
    min_light = 0.0
    max_light = 0.0
    for i in 1:args["episode-length"]*3
        pycall(env.light.step_light, Nothing)
        ff = pycall(env.light.fire_frame, PyArray)
        min_light = min(min_light, minimum(ff))
        max_light = max(max_light, maximum(ff))
    end
    @test min_light == -1
    @test max_light == 1
end
# @testset "test_plot_bcs" begin
#     plot_bcs("$root_dir", Dict(), [[0.99, 0.11], [0.5, 0.5], [0.00, 0.0]])
#     bc_file = read("$root_dir/stats.txt", String) |> split
#     @test bc_file[7] == "0.0"
#     @test bc_file[8] == "0.99"
#     @test bc_file[9] == "0.5"
#     @test bc_file[10] == "0.5"
#     @test bc_file[12] == "0.0"
#     run(`rm $root_dir/stats.txt`)
# end

@testset "test_trade_seed" begin
    expname = ["--exp-name", "test", "--cls-name","test", "--local", "--datime", "test"]
    seed = ["--seed", "123"]
    arg_vector = read("$root_dir/afiles/cls-test/test-ga-trade.arg", String) |> split

    # no seed table
    args = parse_args(vcat(arg_vector, expname), get_arg_table())
    env_config = mk_env_config(args)
    pt = PyTrade()
    env = pt.Trade(env_config)
    @test env isa PyObject
    EvoTrade.Trade.reset!(env)
    no_seed_table = env.table


    args = parse_args(vcat(arg_vector, expname, seed), get_arg_table())
    env_config = mk_env_config(args)
    @test "seed" in keys(env_config)

    env1 = pt.Trade(env_config)
    EvoTrade.Trade.reset!(env1)
    @test env1 isa PyObject
    seed_table_1 = env1.table
    

    env2 = pt.Trade(env_config)
    @test env2 isa PyObject
    EvoTrade.Trade.reset!(env2)
    seed_table_2 = env2.table

    @test seed_table_1 != no_seed_table
    @test seed_table_1 == seed_table_2
end
