struct servoSim
    # Needs to be mainted/configured in Julia:
    beta::Float64
    gain::Float64
    maxTorque::Float64
    rampUpTime::Float64
    dt::Float64
    maxAngle::Float64
    minAngle::Float64
    # Mainted in trianing code:
    prevResidual::Float64
    prevAngle::Float64
    MoI_servo2Rocket::Float64
    MoI_servo2engine::Float64
    desired_angle::Float64
    currAngle::Float64

    
    function servoSim(
        maxAngle,
        minAngle,
        dt,
        rampUpTime,
        maxTorque,
        gain,
        beta
    )
    new(
        maxAngle,
        minAngle,
        dt,
        rampUpTime,
        maxTorque,
        gain,
        beta
    )
    end
end
    
servo1 = servoSim(3.14, 0.0, 0.01, 0.01,2.5,0.5,0.5)

function calc_servo_accel(prevResidual, prevAngle, prevVelocity, I_servo_rocket, I_servo_engine, desired_angle, curr_angle,prev_desired_angle,r1, r2, m1,m2)
    servo1.prevResidual = prevResidual
    servo1.prevAngle = prevAngle
    servo1.MoI_servo2Rocket = I_servo_rocket
    servo1.MoI_servo2engine = I_servo_engine


    #how to characterize ramp up time
    desired_angle_rampup = desired_angle - (prev_desired_angle - desired_angle)/servo1.dt * (servo1.rampUpTime * servo1.dt)
    
    servo1.gain = abs(desired_angle/curr_angle)
    residualMomenta = prevResidual * (servo1.beta) + (1-servo1.beta) * (servo1.currAngle - desired_angle_rampup)
    possible_torque = (desired_angle - prevAngle) * servo1.gain -residualMomenta
    torque = min(possible_torque,servo1.maxTorque)
    I_1 = I_servo_engine + m1 * r1^2
    I_2 = I_servo_rocket + m2 * r2^2
    I_eff = (I_1 * I_2) / (I_1 + I_2)

    servo_accel = torque/I_eff
    velocity = prevVelocity + servo_accel * servo1.dt
    angle = curr_angle + Velocity * servo1.dt
    next_state = Vector{angle,velocity,servo_accel}
    return next_state
    # alpha = torque/((I_servo_engine + I_servo_rocket)/2)
end 