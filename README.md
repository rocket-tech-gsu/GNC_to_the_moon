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
    -Neural networks have made headway in recent years with help of increased levels of computer powers. Many of us know what they are on a high level but to really understand what’s happening requires some necessary insight.

Neural networks are collections of layers of weight matrices and layers of input layers vectors. Where each weight layer corresponding to an N_l row by N_l-1 column weights. The reason for this design is very important. Each input layer is denoted by a set of N scalar variables, where a scalar is 1x1 float number. 

To fully connect every L+1 input node to a combination of each L-1 input node we use simple linear algebra where we left multiply some W(1xN_l) @ X(N_l) = n_l_i
So if we populate a given Hidden Weight layer with N_l+1 of these rows it’d map to N_l+1 scalars. But what makes these helpful for approximating non linear functions is precisely in the activation layers. Activation layers is a 1 to 1 function mapping of each layer’s scalar. Say in the simple example of a ReLu function, which is defined by a a piecewise function: X if X>0 and 0 is X<=0. What makes this non linear is in the transformation and updates that happens to each of these scalars,

 if we were to remove these activation layers. One there’d be no normalization that would keep values within reasonable bounds, causing extreme gradients which can cause instability in our network. Second, each neuron would be a simple scaling operation of the layers before causing updates to be a simple scaling(linear) operation along with backpropogatiom (the updating of the weights by w_t+1 = x 

Let’s make this concrete with an example of our policy network in the soft actor critic case.

 The purpose of neural networks is to approximate a high order function of your choosing, this is the power of NN’s because barring a proper step size you can theoretically update weights to match the objective function. 



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
