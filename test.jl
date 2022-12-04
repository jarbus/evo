using Revise
using EvoTrade

includet("test/test_ga.jl")
includet("test/test_net.jl")
includet("test/test_maze.jl")

function t()
    roc(["test/test_maze.jl"], [EvoTrade]) do
        main()
    end
end
