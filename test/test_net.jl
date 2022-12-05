using EvoTrade
using Test
using Flux


@testset "test_make_model" begin
    for s in [:large, :medium, :small], vbn in [true, false], lstm in [true, false]
        m = make_model(s, (11, 11, 7, 10), 4, vbn=vbn, lstm=lstm)
        out = m(rand(Float32, 11, 11, 7, 10))
        # check that output is correct size
        @test size(out) == (4, 10)
        @test !any(isnan, out)
    end
end
@testset "test_make_head" begin
    for s in [1, 2, 4], vbn in [true, false]
        cnn = EvoTrade.Net.make_head((11, 11, 7, 10), scale=s, vbn=vbn)
        out = cnn(rand(Float32, 11, 11, 7, 10))
        @test size(out, 2) == 10
        @test ndims(out) == 2
        @test !any(isnan, out)
        vbns = vbn ? s - 1 : 0
    end
end
@testset "test_make_tail" begin
    for s in [1, 2, 4], vbn in [true, false], lstm in [true, false]
        input_size = (11, 11, 7, 10)
        cnn = EvoTrade.Net.make_head((11, 11, 7, 10), scale=s, vbn=vbn)
        cnn_size = Flux.outputsize(cnn, input_size)
        tail = EvoTrade.Net.make_tail(cnn_size, 4, scale=s, lstm=lstm)
        cnn_out = cnn(rand(Float32, 11, 11, 7, 10))
        tail_out = tail(cnn_out)
        @test size(tail_out) == (4, 10)
        @test !any(isnan, tail_out)
        @test lstm == any(x->x isa Flux.Recur, tail.layers)
    end
end
