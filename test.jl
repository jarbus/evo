using Revise
using EvoTrade

includet("test/test_ga.jl")

function t()
    roc(["test/test_ga.jl"], [EvoTrade]) do
        main()
    end
end
