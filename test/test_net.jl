using Evo
using Test
using Flux
using PyCall

root_dir = dirname(@__FILE__)  |> dirname |> String
@testset "test_make_model" begin
    for s in [:large, :medium, :small], lstm in [true, false]
        m = make_model(s, (11, 11, 7, 10), 4, lstm=lstm)
        out = m(rand(Float32, 11, 11, 7, 10))
        # check that output is correct size
        @test size(out) == (4, 10)
        @test !any(isnan, out)
    end
end
@testset "test_make_head" begin
    for s in [1, 2, 4]
        cnn = Evo.Net.make_head((11, 11, 7, 10), scale=s)
        out = cnn(rand(Float32, 11, 11, 7, 10))
        @test size(out, 2) == 10
        @test ndims(out) == 2
        @test !any(isnan, out)
    end
end
@testset "test_make_tail" begin
    for s in [1, 2, 4], lstm in [true, false]
        input_size = (11, 11, 7, 10)
        cnn = Evo.Net.make_head((11, 11, 7, 10), scale=s)
        cnn_size = Flux.outputsize(cnn, input_size)
        tail = Evo.Net.make_tail(cnn_size, 4, scale=s, lstm=lstm)
        cnn_out = cnn(rand(Float32, 11, 11, 7, 10))
        tail_out = tail(cnn_out)
        @test size(tail_out) == (4, 10)
        @test !any(isnan, tail_out)
        @test lstm == any(x->x isa Flux.Recur, tail.layers)
    end
end
@testset "test_env_outputs" begin
    expname = ["--exp-name", "test", "--cls-name","test", "--local", "--datime", "test"]
    arg_vector = read("$root_dir/afiles/daystep-test/test-1atrade.arg", String) |> split
    args = parse_args(vcat(arg_vector, expname), get_arg_table())
    env_config = mk_env_config(args)
    action_change = 0
    identical_obs = 0
    for seed in 1:10
        env = PyTrade().Trade(env_config)
        obs = Evo.Trade.reset!(env)
        m = make_model(:large, size(obs["f0a0"]), 4, lstm=true)
        θ, re = Flux.destructure(m)
        mi = ModelInfo(m)
        sc = SeedCache(maxsize=10)
        seeds = [4.0]
        for i in 1:5
            push!(seeds, 1.0, rand(1.0:10.0))
        end
        params = reconstruct(sc, mi, seeds)
        m = re(params)
        prev_out = m(obs["f0a0"])
        prev_obs = obs["f0a0"]
        for i in 1:args["episode-length"]*3
            obs, rew, done = Evo.Trade.step!(env, Dict("f0a0"=>0))
            out = m(obs["f0a0"])
            if argmax(out) != argmax(prev_out)
                action_change += 1
            end
            identical_obs += all(obs["f0a0"][:,:,end-3,1] .== prev_obs[:,:,end-3,1])
            prev_out = out
            prev_obs = obs["f0a0"]
        end
    end
    @test identical_obs == 0
    @test action_change > 1
end
