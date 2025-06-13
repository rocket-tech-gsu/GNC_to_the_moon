import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# 1. Use the continuous version
ENV_ID = "LunarLanderContinuous-v3"

# 2. Vectorized training env
train_env = make_vec_env(ENV_ID, n_envs=4)

# 3. Separate eval env
eval_env = Monitor(gym.make(ENV_ID))

# 4. Callback: stop when mean reward ≥ 200
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=stop_callback,
    eval_freq=10_000,
    best_model_save_path="./logs/",
    verbose=1,
)

# 5. Instantiate SAC
model = SAC(
    policy="MlpPolicy",
    env=train_env,
    verbose=1,
    tensorboard_log="./sac_lunar_tensorboard/",
    batch_size=256,
    learning_rate=3e-4,
)

# 6. Train
model.learn(
    total_timesteps=1_000_000,
    callback=eval_callback
)

# 7. Save the final policy
model.save("sac_lunarlander_continuous_final")

# 8. (Optional) Watch it land
obs, _ = eval_env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = eval_env.step(action)
    eval_env.render()
    if done or truncated:
        obs, _ = eval_env.reset()
