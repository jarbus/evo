include("args.jl")
args = parse_args(s)
include(string("x/"*args["algo"],"trade.jl"))
main()
