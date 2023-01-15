module Net

using Flux
using Statistics
using Functors
using FileIO

export make_model

function gen_temporal_data()
  frame_size = (7, 7, 3, 1)
  num_points = 50
  seq_len = 3
  pos = zeros(Float32, frame_size...)
  pos[:, :, :, :] .= rand()
  neg = zeros(Float32, frame_size...)
  neg[:, :, :, :] .= -rand()
  labels = []
  seq::Vector{Array{Float32,4}} = []
  first_frames = []
  for _ in 1:num_points
    if rand() > 0.5
      push!(first_frames, copy(pos))
      push!(labels, Vector{Float32}([0, 1]))
    else
      push!(first_frames, copy(neg))
      push!(labels, Vector{Float32}([1, 0]))
    end
  end
  push!(seq, cat(first_frames..., dims=4))
  for _ in 1:(seq_len-1)
    push!(seq, zeros(Float32, size(seq[1])))
  end
  seq, hcat(labels...)
end

function make_head(input_size::NTuple{4,Int}; vbn::Bool=false, scale::Int=1)
  layers = Vector{Any}()
  function add_layer(chans, filter, stride)
    vbn && push!(layers, VirtualBatchNorm())
    push!(layers, 
      Conv(filter, chans[1] => chans[2], pad=(1, 1), stride=stride, relu),
    )
  end
  # TODO this is a disgusting hack to avoid parameterizing models based
  # on domain
  if input_size[1] > 20
    add_layer(input_size[3]=>8*scale, (3, 3), 1) # this worked
    add_layer( 8*scale=>16*scale, (8, 8), 4)
    add_layer(16*scale=>16*scale, (4, 4), 2)
    add_layer(16*scale=>16*scale, (3, 3), 1)
  else
    add_layer(input_size[3]=>8*scale, (7, 7), 1)
    add_layer( 8*scale=>16*scale, (5, 5), 1)
    add_layer(16*scale=>32*scale, (3, 3), 1)
  end
  push!(layers, Flux.flatten)
  Chain(layers...)
end

function make_tail(input_size::NTuple{2, Int},
  output_size::Integer;
  lstm::Bool=true,
  scale::Int=1)
    mem = lstm ? LSTM : Dense
    Chain(
        Dense(input_size[1], 64*scale),
        mem(64*scale => 64 * scale),
        relu,
        Dense(64 * scale => 64 * scale, relu),
        Dense(64 * scale => output_size),
        softmax
      )
end

function make_model(size_symbol::Symbol, 
  input_size::NTuple{4,Int},
  output_size::Integer;
  vbn::Bool=false,
  lstm::Bool=true)
    scale_map = Dict(:large=>4, :medium=>2, :small=>1)
    scale = scale_map[size_symbol]
    println("making $(String(size_symbol)) model, lstm=$lstm, vbn=$vbn")
    cnn = make_head(input_size, vbn=vbn, scale=scale)
    cnn_size = Flux.outputsize(cnn, input_size)
    tail = make_tail(cnn_size, output_size, lstm=lstm, scale=4)
    Chain(cnn, tail)
end

function fitness(model; print=false)::Float32
  # y_pred should be vector of probabilities
  x, y_gold = gen_temporal_data()
  Flux.reset!(model)
  [model(xi) for xi in x[1:end-1]]
  y_pred = model(x[end])

  @assert min(y_pred...) >= 0
  @assert max(y_pred...) <= 1
  @assert min(y_gold...) >= 0
  @assert max(y_gold...) <= 1
  @assert size(y_pred) == size(y_gold)

  fit = -Flux.Losses.binarycrossentropy(y_pred, y_gold)

  print && println(" $(round(fit,digits=2))")
  fit
end

function test_lstm()
  m = make_model(:large, (7, 7, 3, 1), 2)
  # x, y = gen_temporal_data()
  # [m(xi) for xi in x[1:end-1]]
  # z = m(x[end])
  fitness(m)
end

end
