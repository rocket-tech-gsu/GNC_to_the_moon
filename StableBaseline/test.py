# check_env.py
import gymnasium as gym
from stable_baselines3 import SAC

# 1) Create the continuous LunarLander
env = gym.make("LunarLanderContinuous-v3")
obs, info = env.reset()
print("Reset successful; obs.shape =", obs.shape)

# 2) Verify SAC can be instantiated
model = SAC("MlpPolicy", env, verbose=0)
print("SAC instantiation successful.")
