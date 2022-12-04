using Revise
using EvoTrade

includet("test/test_ga.jl")
includet("test/test_net.jl")

function t()
    roc(["test/test_ga.jl"], [EvoTrade]) do
        main()
    end
end
