using Revise
using EvoTrade
using Test


function t()
# includet("test/test_ga.jl")
# includet("test/test_net.jl")
    file = "test/test_maze.jl"
    includet(file)
    roc([file], [EvoTrade]) do
        main()
    end
end
