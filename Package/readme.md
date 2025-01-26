I'm documenting my development process here, to understand any code written by me you can just read this readme!
## GPS Simulator:
Parabolic Trajectory Simulation takes only the timestep coming from the time loop.
## Particle Filter:
### Psudo Code:
- Time Loop:
    - state progression
    
    - if time % 5 == 0:
        - update weights 

    - if time % 10 == 0:
        - filter extremes(variation > 3sigma) & nomalize remaining weights
        - kill old particles, populate new particles where the `new particles' probability distribution = old particles' weight distribution`, weight of the new particles can be uniform and normalized