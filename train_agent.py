import mlflow
import os
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from bio_farm_env import BioFarmEnv

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("RL_BioFarm_Optimization")

env = BioFarmEnv()
timesteps = 20000 # Shortened for quick testing

# HYPERPARAMETER TUNING & MULTIPLE MODELS
models_to_test = {
    "PPO_FastLR": {"algo": PPO, "lr": 0.001},
    "PPO_SlowLR": {"algo": PPO, "lr": 0.0003},
    "A2C_Standard": {"algo": A2C, "lr": 0.0007}
}

best_reward = -1000
best_model_name = ""

print("🚀 Starting Automated Hyperparameter Tuning & Model Selection...")

for run_name, config in models_to_test.items():
    with mlflow.start_run(run_name=run_name):
        print(f"Training {run_name}...")
        
        # Log parameters
        mlflow.log_param("Algorithm", config["algo"].__name__)
        mlflow.log_param("Learning_Rate", config["lr"])
        
        # Train
        model = config["algo"]("MlpPolicy", env, learning_rate=config["lr"], verbose=0)
        model.learn(total_timesteps=timesteps)
        
        # Evaluate
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
        mlflow.log_metric("Mean_Reward", mean_reward)
        
        # Save locally to log to MLflow
        model.save("temp_model")
        mlflow.log_artifact("temp_model.zip", artifact_path="models")
        
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_model_name = run_name
            model.save("best_production_model") # Save the winner for the API

print(f"✅ Tuning Complete! Best Model: {best_model_name} (Reward: {best_reward:.2f})")
if os.path.exists("temp_model.zip"): os.remove("temp_model.zip")