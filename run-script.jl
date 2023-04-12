using Evo
args = parse_args(get_arg_table())
make_procs(args)
include(string("x/"*args["algo"],"trade.jl"))
main()
