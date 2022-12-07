using PyCall
root_dir = dirname(@__FILE__)  |> dirname |> String
@testset "test_trade" begin
    expname = ["--exp-name", "test", "--local", "--datime", "test"]
    arg_vector = read("$root_dir/afiles/test-ga-trade.arg", String) |> split
    args = parse_args(vcat(arg_vector, expname), get_arg_table())
    env_config = mk_env_config(args)
    env = PyTrade().Trade(env_config)
    @test env isa PyObject
end

@testset "test_plot_bcs" begin
    println("fsdfdsf")
    plot_bcs("$root_dir", [[0.9, 0.1], [0.5, 0.5]])
    bc_file = read("$root_dir/stats.txt", String) |> split
    @test bc_file[7] == "0.5"
    @test bc_file[8] == "0.9"
    @test bc_file[9] == "0.7"
    run(`rm $root_dir/stats.txt`)
end
