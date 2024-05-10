using TOML

function collectsysteminfo(fn = "systeminfo.toml")
    cpu = IOBuffer()
    Sys.cpu_summary(cpu)
    cpu = String(take!(cpu))

    gpu = CUDA.name( CUDA.device() )
    open(fn, "w") do f 
        TOML.print(f, Dict(:cpu => cpu, :gpu => gpu))
    end
end