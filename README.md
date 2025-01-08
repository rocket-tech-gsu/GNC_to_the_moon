# GNC_to_the_moon
In-house developed Guidance Navigation Control software libraries!

# Goal of this project:
- Write accurate analytically derived numerical simulation for the system dynamics for accurate dead reckonning in case of abnormal behavior of the state estimation system.

For a given set of initial conditions:
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