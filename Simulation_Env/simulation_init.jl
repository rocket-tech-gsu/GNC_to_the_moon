"""
This file contains the initialization code for a rocket and the flight environment.
It's a completely configurable environment:
- Rocket's Configuration:
  - Either Electric Thrust Configuration(that uses a propellor and a motor)
  - Or a Solid Rocket Enginge
- Rocket's Parameters:
  - 
- Sensor Configuration:
  - Select the sensors to be simulated on board.
- Sensor Parameters
  - Initialize selected sensors with either default or custom parameters.
  - Example:
    {
        GPS: {std, refresh_rate, {upperbounds:velocity, acceleration}, Delay},
        Pressure: {std, refresh_rate, Delay},

    }
- Actuator Configuration
  -  
- Actuator Parameters
"""
using Distributions
using StaticArrays
using CSV, DataFrames
using Random, Distributions
@enum flight_configuration drone_lift two_staged
@enum control_configuration two_dof_tvc three_dof_tvc no_control # 3DOF tvc = 2DOF tvc + throttling

# Everything must be in SI units(say no to emperical)
struct PhysicalParam
    fluctuate::Bool
    value::Float64
end

function randomize(Rocket::PhysicalParams_Solid, seed)
    Random.seed!(seed)
    # Randomizing State Vector:
    
    # Randomizing Physical Parameters of the Rocket:
    length =  Normal(Rocket.length, physical_params_std)
    diameter = Normal(Rocket.diameter, physical_params_std)
    cop = Normal(Rocket.cop, physical_params_std)
    mass_rocket_dry = Normal(Rocket.mass_rocket_dry, physical_params_std)
    com_dry = Normal(Rocket.com_dry, physical_params_std)
    engine_location = Normal(Rocket.engine_location, physical_params_std)
    
    return PhysicalParams_Solid(length, diameter, cop, mass_rocket_dry, com_dry, engine_location, 
    Rocket.mass_engine, Rocket.inertia_roll_dry, Rocket.inertia_pitch_dry, Rocket.throttle)

end


# Read the CSV file into a DataFrame
df = CSV.read("/Users/varunahlawat/Downloads/Motor_Simulation_data.csv", DataFrame)
println(df)
# Extract a column as a Julia vector (list)
thrust = df[!, "Thrust(N)"]
m1 = df[!, "Propellant Mass(G1;g)"]  
m2 = df[!, "Propellant Mass(G2;g)"]
m3 = df[!, "Propellant Mass(G3;g)"]



