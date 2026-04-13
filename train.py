from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import CreatureEnv

# 10 agents
env = DummyVecEnv([lambda: CreatureEnv() for _ in range(10)])

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=200000)

model.save("crawler_model")

print("Training done")