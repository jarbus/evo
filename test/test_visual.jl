using Evo
using Test
using PyCall
using Plots

# TODO FIX THIS TEST
# @testset "test_trade_pop_visualizer" begin
#     root_dir = dirname(@__FILE__)  |> dirname |> String
#     expname = ["--exp-name", "test", "--cls-name","test", "--local", "--datime", "test"]
#     seed = ["--seed", "123"]
#     arg_vector = read("$root_dir/afiles/cls-test/test-ga-trade.arg", String) |> split
#
#     args = parse_args(vcat(arg_vector, expname), get_arg_table())
#     env_config = mk_env_config(args)
#     pt = PyTrade()
#     env = pt.Trade(env_config)
#     @test env isa PyObject
#     Evo.reset!(env)
#     walks = [[(4,4), (3,4), (3,3), (3,2), (2, 2), (1,2), (1,3), (1,4), (1,5)]]
#     plot_walks("walks.png", env.table, walks)
# end

# @testset "plot_rewbcs" begin
#     inds = [Ind([1f0]) for i in 1:10]
#     for ind in inds
#         ind.bc = rand(Float32, 9)
#         ind.novelty = 1f0
#     end
#     pop = Pop("1", 10, inds)
#     push!(pop.archive, rand(Float32, 9))
#     Evo.plot_rewbcs("bcs.png", pop, 3)
#     @test true
# end


@testset "plot_5bcs" begin
    inds = [Ind([1f0]) for i in 1:10]
    for ind in inds
        ind.bc = rand(Float32, 5)
        ind.novelty = 1f0
    end
    pop = Pop("1", 10, inds)
    push!(pop.archive, rand(Float32, 5))
    Evo.plot_bcs("bcs", [pop], 3)
    @test true
end

#@testset "test_trade_pop_visualizer" begin
#    root_dir = dirname(@__FILE__)  |> dirname |> String
#    expname = ["--exp-name", "test", "--cls-name","test", "--local", "--datime", "test"]
#    seed = ["--seed", "123"]
#    arg_vector = read("$root_dir/afiles/cls-test/test-ga-trade.arg", String) |> split
#
#    args = parse_args(vcat(arg_vector, expname), get_arg_table())
#    env_config = mk_env_config(args)
#    pt = PyTrade()
#    env = pt.Trade(env_config)
#    @test env isa PyObject
#    Evo.reset!(env)
#    moves = env.MOVES
#    println(moves)
#    @test env.agent_positions["f0a0"] == (4,4)
#    idx = findfirst(x->x=="UP", moves)-1
#    Evo.step!(env, Dict("f0a0"=>idx))
#end


