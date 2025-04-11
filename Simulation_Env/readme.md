**Simulation_Env** contains the entire Julia Simulation code that has the following functionalities:

- Solving Dynamics
- Simulating:
  - Individual Parts of Flights like landing or takeoff
  - Complete Flight


# Setup:
Make sure your computer has julia installed. 
Run the following commands in the terminal(make sure you're in the right directory).
```
Julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```