import gym
from stable_baselines3 import A2C
import os
import time

models_dir = f"models/A2C-{int(time.time())}"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = gym.make('LunarLander-v2')
env.reset()

# multilayered perceptron = MlpPolicy
model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
TIMESTEPS = 10000

for i in range(1,100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"{models_dir}/{TIMESTEPS*i}")