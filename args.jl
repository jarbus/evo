using ArgParse
s = ArgParseSettings()
@add_arg_table! s begin
  "--local"
  action = :store_true
  "--window"
  nargs = 2
  arg_type = Int
  "--gx"
  arg_type = Int
  "--gy"
  arg_type = Int
  "--fires"
  nargs = '*'
  arg_type = Int
  "--foods"
  nargs = '*'
  arg_type = Int
  "--exp-name"
  arg_type = String
  "--class-name"
  arg_type = String
  "--model"
  arg_type = String
  "--num-gens"
  arg_type = Int
  "--day-steps"
  arg_type = Int
  "--batch-size"
  arg_type = Int
  "--episode-length"
  arg_type = Int
  "--num-agents"
  arg_type = Int
  "--pop-size"
  arg_type = Int
  "--alpha"
  arg_type = Float32
  "--mutation-rate"
  arg_type = Float32
  "--num-steps"
  "--checkpoint-interval"
  "--food-agent-start"
  "--food-env-spawn"
  "--light-coeff"
  "--pickup-coeff"
  "--food-types"
end
args = parse_args(s)
