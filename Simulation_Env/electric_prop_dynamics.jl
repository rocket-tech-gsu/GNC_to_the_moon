using LinearAlgebra
using StaticArrays
# Dynamics Solver Parameters
g = 9.81    # m/(second)^2
# dt = 1/30   # seconds

struct PhysicalParam
    fluctuate::Bool
    value::Float64
end

mutable struct PhysicalParams_Electric # Solid Rocket Engine
    # Fixed Parameters:
    length::PhysicalParam
    diameter::PhysicalParam
    cop::PhysicalParam
    mass_rocket_dry::PhysicalParam
    com_dry::PhysicalParam
    engine_location::PhysicalParam # Engine's com distance from the Rocket's dry com
    # Variable Parameters(due to varying mass):
    com::Vector{Float64}
    inertia_roll_dry::PhysicalParam
    inertia_pitch_dry::PhysicalParam
    throttle::Bool
    inertia_roll::Float64
    inertia_pitch::Float64  # Dry Rocket About Pitch/Yaw Axis passing through COM(t) 
    inertia_com_rocket_dry_pitch::Float64
    inertia_engine_rockets_com_pitch::Float64
    mass_engine::Union{Float64, Nothing} # Will come from Engine Profile
    thrust::Union{Vector{Float64}, Nothing}
    
    # Solid Propellant Variables
    prop_grain_inner_dia::Union{PhysicalParam, Nothing}
    prop_grain_outer_dia::Union{PhysicalParam, Nothing}
    prop_grain_length::Union{Float64, Nothing}
    prop_grain_density::Union{Float64, Nothing}

    function PhysicalParams_Electric(
        length::PhysicalParam,
        diameter::PhysicalParam,
        cop::PhysicalParam,
        mass_rocket_dry::PhysicalParam,
        com_dry::PhysicalParam,
        engine_location::PhysicalParam, # Measured from the dry COM
        inertia_roll_dry::PhysicalParam,
        inertia_pitch_dry::PhysicalParam,
        throttle::Bool,
        mass_engine::Union{Float64, Nothing} = nothing,
        thrust::Union{Vector{Float64}, Nothing} = nothing
    )
        if throttle == true
            thrust = 0
            println("... Throttling Activated ...")
        else
            if isnothing(thrust)
                println("Populate Thrust Profile!")
            end
        end
        
        # Create a proper PhysicalParam object for r_prop_com
        r_prop_com = PhysicalParam(true, (diameter.value/2) * 0.5)
        
        # Dynamically Calculating COM, Moment of Inertias:
        # Vector operations need dots
        com = (engine_location.value) * ((mass_engine) / (mass_engine + (mass_rocket_dry.value)))
        
        # Roll Inertia:
        inertia_roll = (inertia_roll_dry.value) + ((mass_engine) * (r_prop_com.value * r_prop_com.value))
        
        # Rocket's Pitch/Yaw Inertia:
        inertia_com_rocket_dry_pitch = (inertia_pitch_dry.value) + (mass_rocket_dry.value * ((engine_location.value * (mass_engine / (mass_rocket_dry.value + mass_engine)))^2)) # about the new com pitch axis
        inertia_engine_rockets_com_pitch = (engine_location.value^2) * (mass_engine * (mass_rocket_dry.value/(mass_rocket_dry.value + mass_engine))^2)
        inertia_pitch = inertia_com_rocket_dry_pitch + inertia_engine_rockets_com_pitch    # Parallel Axis Theorem -> Accounting for shifting COM axis.
        
        # Engine's Pitch/Yaw Inertia:
            # prop_grain_inner_dia
            # prop_grain_outer_dia
            # prop_grain_length
            # prop_grain_density
        inertia_engine_wet_pitch = 

        # constructor
        new(
            length,
            diameter,
            cop,
            mass_rocket_dry,
            com_dry,
            engine_location,
            com,
            inertia_roll_dry,
            inertia_pitch_dry,
            throttle,
            inertia_roll,
            inertia_pitch,
            inertia_com_rocket_dry_pitch,
            inertia_engine_rockets_com_pitch,
            mass_engine,
            r_prop_com
            # thrust
        )
    end
end

function R2U_rotation(vec_in, a, b, c)
    Rx_matrix = SMatrix{3,3}(
        1,        0,      0,
        0,   cos(a),-sin(a),
        0,  sin(a), cos(a),
        )

    Ry_matrix = SMatrix{3,3}(
        cos(b), 0,  sin(b),
            0, 1,        0,
        -sin(b), 0,   cos(b),
    )

    Rz_matrix = SMatrix{3,3}(
        cos(c), -sin(c), 0,
        sin(c),  cos(c), 0,
            0,       0, 1,
    )
    R2U_matrix = (Rx_matrix * Ry_matrix * Rz_matrix)'
    vec_out = R2U_matrix * vec_in
    return vec_out
end

function safe_divide(numerator, denominator; tol=1e-15)
    # Check if the denominator is effectively zero
    if abs(denominator) < tol
        error("Denominator is too close to zero!")
    end
    return numerator / denominator
end

function Electric_prop_dynamics(rocket::PhysicalParams_Electric, t, state_vector, actuator_state)
    """
    For a given rocket data object, state vector, actuator state, time t this function simulates, returns the rocket's state vector at t+1 step.
    """
    theta_1 = actuator_state[0]
    theta_2 = actuator_state[1]
    e_vec = [sin(theta_2) * cos(theta_1),   # unit vector in the direction of exhaust plume
    sin(theta_2) * sin(theta_1),
    cos(theta_2)
    ]
    thrust = rocket.thrust[t]
    thrust_vec_R = thrust * e_vec
    thrust_vec_U = R2U_rotation(thrust_vec_R, 
                                state_vector[10], 
                                state_vector[11], 
                                state_vector[12])

    # CFD LookUp Table Aerodynamics
    # F_aerodynamics = 

    # Linear Accelerations:
    a1 = thrust_vec_U[0]
    a2 = thrust_vec_U[1]
    a3 = thrust_vec_U[2] + g
    
    # Angular Momentum Conservation Angular Velocity Correction:
    W_r = I_ratio .* R2U_rotation()
    W1 = W_r[0]
    W2 = W_r[1]
    
    # TODO: Angular Accelerations:
    A1 = 1
    A2 = 1
    A3 = 1

    # A Matrix (TODO: Use SparseArrays)
    A = SMatrix{18,18}(
        1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # linear accelerations
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 + W1, 0, 0, dt, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 + W2, 0, 0, dt, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt,
        # angular accelerations
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    )
    tmp_state = A * state_vector

    tmp_state[6] = a1
    tmp_state[7] = a2
    tmp_state[8] = a3

    tmp_state[end-2] = A1
    tmp_state[end-1] = A2
    tmp_state[end] = A3
    return (tmp_state)

end
