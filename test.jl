using Revise
using EvoTrade
using Test


function t()
    # file = "test/test_ga.jl"
    # file = "test/test_net.jl"
    # file = "test/test_maze.jl"
    # file = "test/test_trade.jl"
    file = "test/test_rollout.jl"
    includet(file)
    main()
    roc([file], [EvoTrade]) do
        println("Running main()")
        main()
    end
end
t()
