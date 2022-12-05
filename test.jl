using Revise
using EvoTrade
using Test


function t()
    # file = "test/test_ga.jl"
    # file = "test/test_net.jl"
    # file = "test/test_maze.jl"
    file = "test/test_trade.jl"
    includet(file)
    roc([file], [EvoTrade]) do
        println("Running main()")
        main()
    end
end
t()
