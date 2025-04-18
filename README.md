# GNC_to_the_moon
In-house developed Guidance, Navigation, and Control software libraries! 

### 1. Guidance
This piece of software will be able to calculate optimal trajectories for different missions(both orbital and non-orbital). We will call its final output as the ___desired path___.<br>
<u>__Note:__</u>These calculations will be done on the ground systems and not on the avionics. However, the output will be statically fed in the onboard avionics memory before the mission.

### 2. Navigation
This piece of software will be using the noisy, meaningless voltage values from onboard sensors and will be able to reproduce a sense of current position and orientation during flight. 
<br>
- It'll include a sensor fusion algorithm to fight noise(like `Band Pass Filter`) at an individual sensor level. 
- For state estimation currently we have decided to develop `Particle Filter` algorithm.
<br>
<u>__Note:__</u> Unlike gaussian assumption based kalman filters, particle filter algorithm is more generealized and adaptable.

### 3. Control
The whole system should then be able to conduct a mission autonomously. Goal of this software will be to dynamically generate control values for actuators(to control the thrust direction) keep/bring the rocket close to the ___desired path___.

We have opted to develop Model `Predictive Control Algorithm`. 
However there're three parts to it:

- __Analytically Modelling the System Dynamics:__
    - Newtonian Physics : Independent variables(updating from Navigation code).
    - OpenFOAM Computational Fluid Dynamics(CFD) : CFD simulations on the 3D model to get $C_D$ values for a grid of different($\vec{V}, \hat{n},\rho_{air}$) values.
- __Model Predictive Control:__
  - Linear Algebraic structures abstracts away all the system dynamics and helps us calculate a loss function between desired and the (current + predicted) trajectory. It then spits out control values which can then minimize the loss function.
<br><u>Note:</u> Unlike the heuristical control algorithms like PID we opted `Model Predictive control`, which is an optimal control algorithm(meaning instead of sticking to a fixed logic, it can adapt the itself to the changing environment) 

- __Reinforcement Learning:__ 
  - In MPC, the weight matrices of the loss function are not analytically derived. Instead, they're traditionally populated with hit and try values, which are quite low(only because there're a high number of values that contributes to the loss so individual contributions needs to be quite small).
  - These weight values clearly affect the control performance. Hence good Reinforcement Learning can tune these weights. It will really make a difference! 
  However, that will be a naieve thing to do. Why, you might ask! If the underlying system modeling is the limiting factor for your control system then the weight matrices of the loss function cannot help much! At max you can tune the tradeoff between:
    - Higher position weights (Q) → Better tracking but more aggressive control
    - Higher input weights (R) → Smoother control but worse tracking
    To be fair, the results of this approach are unknown because no one had ever tried it before!

Deep Learning Basics: 

In this implementation, we chose to use a deep reinforcement learning algorithm called Soft Actor Critic. A main factor that makes soft actor critic special is the use of the -log(P(x)) in it's loss function. In reinforcement learning it is standard that instead of minimizing some function you maximize some function! The function that we want to maximize is denoted as this the Avg of (Q(s_t) - log(a_t)) Despite the sign of this formula, this number for the range of output probabilites of [0-1] scales Higher to Lower, where -log(1)=0 and as you get to a lower probability action, the function. So you can imagine that during our training we store a state tuple within our replay buffer ()

## Overall Goal:
Write accurate analytically derived numerical simulation for the system dynamics for accurate dead reckonning in case of abnormal behavior of the state estimation system.

<u>For a given set of initial conditions:</u>
- The State of the Rocket should be accurately and precisely measured at every time step.
- Guidance should be generated dynamically for a specifically designed mission. Unexpected conditions should be handled.
- The control system should be able to send actuation values to the thrust vectoring system in order to send a rocket to the moon.

- Implement an Actor-Critic reinforcement learning based pipeline to tune the model control algorithm for a given rocket and a mission.
- Data from the real world static propulsion test setups and real test flights should be able to improve the whole model and potentially the pipeline.

- The live computations onboard must not be too complicated, anything that ESP32 may not be able to do.

# Setup:
There are some dependencies in requirements.txt that needs an older version of python: `3.12.7`
- Make a virtual environment using that version of python using:
```
python3.12 -m venv env
```
- Activate that virtual environment:
for macOS users:
```
source env/bin/activate
```
for Windows users:
```
source env/scripts/activate
```
- Install the dependencies in this virtual environment now:
```
pip install -r requirements.txt
```
# Open Source Contribution Instructions:
Before you stage all the changes for the final time just before the pull request, make sure to update any new packages in the requirements.txt file using the following command:
```
pip freeze > requirements.txt
```
