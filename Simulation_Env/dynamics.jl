using LinearAlgebra
using StaticArrays
# Dynamics Solver Parameters
g = 9.81    # m/(second)^2
# dt = 1/30   # seconds

struct PhysicalParam
    fluctuate::Bool
    value::Float64
end

mutable struct PhysicalParams_Solid    # Solid Rocket Engine
    # Fixed Parameters:
    length::PhysicalParam
    diameter::PhysicalParam
    cop::PhysicalParam
    mass_rocket_dry::PhysicalParam
    com_dry::PhysicalParam
    engine_location::PhysicalParam  # Engine's com distance from the Rocket's dry com

    # Variable Parameters(due to varying mass):
    com::Vector{Float64}
    inertia_roll_dry::PhysicalParam
    inertia_pitch_dry::PhysicalParam
    
    throttle::Bool
    
    inertia_roll::Vector{Float64}
    inertia_pitch::Vector{Float64}
    inertia_com_rocket_t::Vector{Float64}
    inertia_engine_com_t::Vector{Float64}
    
    mass_engine::Union{Vector{Float64}, Nothing} # Will come from Engine Profile
    r_prop_com::Union{PhysicalParam, Nothing}
    thrust::Union{Vector{Float64}, Nothing}
    function PhysicalParams_Solid(length::PhysicalParam,
                                    diameter::PhysicalParam,
                                    cop::PhysicalParam,
                                    mass_rocket_dry::PhysicalParam,
                                    com_dry::PhysicalParam,
                                    engine_location::PhysicalParam, # Measured from the dry COM
                                    inertia_roll_dry::PhysicalParam,
                                    inertia_pitch_dry::PhysicalParam,
                                    throttle::Bool,
                                    mass_engine::Union{Vector{Float64}, Nothing} = nothing,
                                    thrust::Union{Vector{Float64}, Nothing} = nothing
                                )
        if throttle==true
            thrust = 0
        else
            if isnothing(thrust)
                println("Populate Thrust Profile!")
            end
        end

        r_prop_com = (diameter.value/2) * 0.5   # Mass of propellant could be thought of concentrated at this radius

        # Dynamically Calculating COM, Moment of Inertias:     
        
        # Vector operations need dots
        com = (engine_location.value) .* ((mass_engine) ./ (mass_engine .+ (mass_rocket_dry.value)))
        # Roll Inertia:
        inertia_roll = (inertia_roll_dry.value) .+ ((mass_engine) .* (r_prop_com * r_prop_com))
        # Pitch/Yaw Inertia:
        inertia_com_rocket_t = (inertia_pitch_dry.value) .+ (mass_rocket_dry.value .* ((engine_location.value .* (mass_engine ./ (mass_rocket_dry.value + mass_engine))).^2)) # about the new com pitch axis
        inertia_engine_com_t = (engine_location.value^2) .* (mass_engine .* (mass_rocket_dry.value./(mass_rocket_dry.value .+ mass_engine)).^2)
        inertia_pitch = inertia_com_rocket_t .+ inertia_engine_com_t

        # constructor
        new(length, diameter, cop, mass_rocket_dry, com_dry, engine_location,
            com, inertia_roll_dry, inertia_pitch_dry, throttle, inertia_roll,
            inertia_pitch, inertia_com_rocket_t, inertia_engine_com_t, mass_engine, r_prop_com, thrust)
    end
end


function R2U_rotation(vec_in, a, b, c)
    Rx_matrix = @SMatrix [
        1,        0,      0;
        0,   cos(a),-sin(a);
        0,  sin(a), cos(a);
        ]

    Ry_matrix = @SMatrix [
        cos(b), 0,  sin(b);
            0, 1,        0;
        -sin(b), 0,   cos(b);
    ]

    Rz_matrix = @SMatrix [
        cos(c), -sin(c), 0;
        sin(c),  cos(c), 0;
            0,       0, 1;
    ]
    R2U_matrix = (Rx_matrix * Ry_matrix * Rz_matrix)'
    vec_out = R2U_matrix * vec_in
    return vec_out
end


function Dynamics(rocket::PhysicalParams_Solid, t, state_vector, actuator_state)
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

    # parsing current time step's(t) values:
    a1_old = state_vector[7]
    a2_old = state_vector[8]
    a3_old = state_vector[9]
    A1_old = state_vector[16]
    A2_old = state_vector[17]
    A3_old = state_vector[18]

    F_aerodynamics = 

    # Linear Accelerations:
    a1 = thrust_vec_U[0]
    a2 = thrust_vec_U[1]
    a3 = thrust_vec_U[2] + g
    
    # TODO: Compute Engine's Pitch Axis Moment of Inertia from Roll Aixs' moment of inertia
    inertia_roll = rocket.inertia_engine_com_t
    inertia_pitch = 
    
    # Angular Momentum Conservation Angular Velocity Correction:
    W_r = I_ratio .* R2U_rotation()
    W1 = W_r[0]
    W2 = W_r[1]
    
    # TODO: Angular Accelerations:
    A1 = 1
    A2 = 1
    A3 = 1

    # A Matrix
    A = [1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        # linear accelerations
        0, 0, 0, 0, 0, 0, a1/a1_old, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0, 0, a2/a2_old, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 0, a3/a3_old, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 + W1, 0, 0, dt, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 + W2, 0, 0, dt, 0;
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt;
        # angular accelerations
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, A1/A1_old, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, A2/A2_old, 0;
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, A3/A3_old;
    ]
    return (A * state_vector)

end
