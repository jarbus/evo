using Distributed
if !args["local"]
  function get_procs(str)
    full_cpus_per_node = Vector{Int}()
    for c in str
      m = match(r"(\d+)\(x(\d+)\)", c)
      if !isnothing(m)
        for i in 1:parse(Int, m[2])
          push!(full_cpus_per_node, parse(Int, m[1]))
        end
      else
        push!(full_cpus_per_node, parse(Int, c))
      end
    end
    full_cpus_per_node
  end

  cpus_per_node = get_procs(split(ENV["SLURM_JOB_CPUS_PER_NODE"], ","))
  nodelist = ENV["SLURM_JOB_NODELIST"]
  hostnames = read(`scontrol show hostnames "$nodelist"`, String) |> strip |> split .|> String
  @assert length(cpus_per_node) == length(hostnames)

  machine_specs = [hostspec for hostspec in zip(hostnames, cpus_per_node)]
  println(machine_specs)
  addprocs(machine_specs, max_parallel=100, multiplex=true)
  println("nprocs $(nprocs())")
else
  addprocs(1)
end
