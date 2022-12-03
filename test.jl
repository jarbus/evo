using Revise
using EvoTrade

includet("EvoTrade/test/test_ga.jl")

function t()
    roc(["EvoTrade/test/test_ga.jl"], [EvoTrade]) do
        main()
    end
end
