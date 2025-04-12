"""
This file contains the Main Simulation Loop!
"""

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using CSV, DataFrames
using Random, Distributions
# include("dynamics.jl")
# include("simulation_init.jl")

# Read the CSV file into a DataFrame
df = CSV.File("/Users/varunahlawat/Downloads/Motor_Simulation_data.csv") |> DataFrame
# Extract a column as a Julia vector (list)
thrust = df."Thrust(N)"
# println("THRUST ?",thrust)

m1 = df[!, "Propellant Mass(G1;g)"]
m2 = df[!, "Propellant Mass(G2;g)"]
m3 = df[!, "Propellant Mass(G3;g)"]
mass = ((m1 .+ m2 .+ m3) ./ 1000)
println("\nWet Mass(in Kg) Timeseries :\n",thrust)

Rocket1 = PhysicalParams_Solid(
    PhysicalParam(true, Float64(1.5)),      # length
    PhysicalParam(true, Float64(0.05)),     # diameter
    PhysicalParam(true, Float64(0.8)),      # cop
    PhysicalParam(true, Float64(2.5)),      # mass_rocket_dry
    PhysicalParam(true, Float64(0.4)),      # com_dry
    PhysicalParam(true, Float64(1.35)),     # engine_location
    PhysicalParam(true, Float64(0.5)),      # inertia_roll_dry
    PhysicalParam(true, Float64(2.0)),      # inertia_pitch_dry
    false,                                  # throttle
    Float64.(mass),                     # mass_engine
    Float64.(thrust)                        # thrust
)

# # Usage
# function Simulation(seed::Int)
#     Rocket_random = randomize(Rocket1)

#     # Print the first 10 samples to verify
#     # Simulation Parameters:
#     N_episodes = 1000
#     for simulation in 1:N_episodes
#         """
#         Simulation Function call will be done in Python -> Parameters will be coming from Python's call
#         Args:
#         - Action
#         - Previous State
#         Return(inputs for the RL code):
#             - RocketState + ActuatorState
#         """
#         done = false
#         while(!done)
#             done = true
#         end
#     end
# end
