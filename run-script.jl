include("args.jl")
args = parse_args(s)
include(string(args["algo"],"trade.jl"))
main()
