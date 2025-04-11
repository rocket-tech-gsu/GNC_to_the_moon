import StaticArrays

mutable struct sensor_suite
    accelerometers::SVector{3,Float64}
    gyroscopes::SVector{3,Float64}
    magnetometers::SVector{3,Float64}
    gps::SVector{3,Float64}
    sensor_state::SVector{12,Float64}

    function SensorSuite(acc,gyro,mag,gps)
        sensor_state = vcat(acc, gyro, mag, gps)
        new(acc, gyro, mag, gps, sensor_state)
    end
end