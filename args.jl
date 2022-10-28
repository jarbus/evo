using ArgParse
s = ArgParseSettings()
@add_arg_table! s begin
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
  "--class-name"
  arg_type = String
  "--model"
  arg_type = String
  "--num-gens"
  arg_type = Int
  "--day-steps"
  arg_type = Int
  "--episode-length"
  arg_type = Int
  "--num-agents"
  arg_type = Int
  "--pop-size"
  "--num-steps"
  "--checkpoint-interval"
  "--food-agent-start"
  "--food-env-spawn"
  "--light-coeff"
  "--pickup-coeff"
  "--food-types"
end
file_args = read("2p-large-1.arg", String) |> split
args = parse_args(file_args, s)
