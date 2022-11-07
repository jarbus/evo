using ArgParse
s = ArgParseSettings()
@add_arg_table! s begin
  "--local"
  action = :store_true
  "--window"
  nargs = 2
  arg_type = Int
  required = true
  "--gx"
  arg_type = Int
  required = true
  "--gy"
  arg_type = Int
  required = true
  "--fires"
  nargs = '*'
  arg_type = Int
  required = true
  "--foods"
  nargs = '*'
  arg_type = Int
  required = true
  "--exp-name"
  arg_type = String
  required = true
  "--model"
  arg_type = String
  required = true
  "--num-gens"
  arg_type = Int
  required = true
  "--day-steps"
  arg_type = Int
  required = true
  "--batch-size"
  arg_type = Int
  required = true
  "--episode-length"
  arg_type = Int
  required = true
  "--num-agents"
  arg_type = Int
  required = true
  "--pop-size"
  arg_type = Int
  required = true
  "--alpha"
  arg_type = Float32
  required = true
  "--l2"
  arg_type = Float32
  required = true
  "--mutation-rate"
  arg_type = Float32
  required = true
  "--food-env-spawn"
  arg_type = Int
  required = true
  "--food-agent-start"
  arg_type = Int
  required = true
  "--num-steps"
  "--checkpoint-interval"
  "--light-coeff"
  "--pickup-coeff"
  "--food-types"
  "--class-name"
end
