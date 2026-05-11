import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Look inside the 'src' folder for the environment
from src.bio_farm_env import BioFarmEnv

env = BioFarmEnv()

print("🧠 Initializing Deep Neural Network...")
model = PPO("MlpPolicy", env, verbose=1)

print("🚀 Training the PPO Agent (50,000 timesteps)...")
model.learn(total_timesteps=50000)

# Save the model directly into the 'models' folder
model_path = "models/ppo_model"
model.save(model_path)
print(f"✅ Model saved successfully to {model_path}.zip")

print("\n📊 Evaluating the Trained AI...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean Final Reward: {mean_reward:.2f} +/- {std_reward:.2f}")