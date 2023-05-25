import gym
from stable_baselines3 import PPO, A2C

env = gym.make("LunarLander-v2")

models_dir = f"models/PPO-1684919758"
model_path = f"{models_dir}/850000.zip"

model = PPO.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        #model predict returns states as well
        action, _ = model.predict(obs)
        obs,reward,done,info = env.step(action)
    env.close()