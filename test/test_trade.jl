using PyCall
@testset "test_trade" begin
    expname = ["--exp-name", "test", "--local", "--datime", "test"]
    arg_vector = read("afiles/test-ga-trade.arg", String) |> split
    args = parse_args(vcat(arg_vector, expname), get_arg_table())
    env_config = mk_env_config(args)
    env = PyTrade().Trade(env_config)
    @test env isa PyObject
end