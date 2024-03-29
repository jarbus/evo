function get_arg_table()
    s = ArgParseSettings()
    @add_arg_table! s begin
      "--food-types"
      arg_type = Int
      required = true
      "--datime"
      arg_type = String
      required = true
      "--rollout-group-size"
      arg_type = Int
      required = true
      "--rollout-groups-per-mut"
      arg_type = Int
      required = true
      "--algo"
      arg_type = String
      required = true
      "--exploration-rate"
      arg_type = Float64
      required = true
      "--local"
      action = :store_true
      "--lstm"
      action = :store_true
      "--maze"
      arg_type = String
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
      "--cls-name"
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
      required = false
      "--pop-size"
      arg_type = Int
      required = true
      "--num-elites"
      arg_type = Int
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
      "--light-coeff"
      arg_type = Float32
      required = true
      "--pickup-coeff"
      arg_type = Float32
      required = true
      "--archive-prob"
      arg_type = Float32
      required = true
      "--seed"
      arg_type = Int
      "--checkpoint-interval"
      "--class-name"
      "--nprocs"
    end
end
