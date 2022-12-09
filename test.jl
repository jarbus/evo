using Revise
using ProfileView
using BenchmarkTools
using EvoTrade
using StableRNGs
using Test


file = "test/test_noisetable.jl"
# file = "test/test_ga.jl"
# file = "test/test_net.jl"
# file = "test/test_maze.jl"
# file = "test/test_trade.jl"
# file = "test/test_rollout.jl"
includet(file)
function t()
    roc([file], [EvoTrade]) do
        println("Running main()")
        main()
    end
end
