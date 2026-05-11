import streamlit as st
import numpy as np
import time
import os
from stable_baselines3 import PPO

# ✅ TWEAK 1: Importing the environment from your new 'src' folder
from src.bio_farm_env import BioFarmEnv

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="RL Bio-Farm Optimizer", layout="wide")
st.title("🌱 RL-Powered Bio-System Dashboard")
st.markdown("Observe the Reinforcement Learning agent autonomously manage resource consumption.")

# --- SIDEBAR: MODEL MANAGEMENT ---
st.sidebar.header("RL Configuration")

# TWEAK 2: The model path now points strictly to the 'models' folder
MODEL_PATH = "models/ppo_model"
MODEL_FILE = f"{MODEL_PATH}.zip"

if st.sidebar.button("Train New PPO Agent"):
    with st.spinner("Training RL Model (50,000 timesteps)... This will take a moment."):
        env = BioFarmEnv()
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=50000)
        
        # Save the brain safely in the models directory
        model.save(MODEL_PATH)
        st.sidebar.success("✅ Training Complete!")

# Check if the AI brain actually exists before letting the user run the simulation
if os.path.exists(MODEL_FILE):
    st.sidebar.success("✅ PPO Model Ready for Simulation")
else:
    st.sidebar.error("❌ No model found. Please click 'Train New PPO Agent' first.")
    st.stop()

# --- MAIN DASHBOARD: SIMULATION ---
if st.button("🚀 Run 60-Day RL Simulation Cycle"):
    # Initialize environment and load the trained brain
    env = BioFarmEnv()
    model = PPO.load(MODEL_PATH)
    obs, _ = env.reset()
    
    st.subheader("Live Sensor Data")
    
    # Create empty UI columns for the dashboard
    col1, col2, col3, col4, col5 = st.columns(5)
    growth_metric = col1.empty()
    moisture_metric = col2.empty()
    nutrient_metric = col3.empty()
    temp_metric = col4.empty()
    light_metric = col5.empty()

    st.subheader("Agent Action Log")
    log_window = st.empty()
    log_text = ""
    
    total_cost = 0
    
    # The 60-Day Simulation Loop
    for day in range(60):
        # 1. The AI looks at the sensors and chooses an action
        action, _states = model.predict(obs, deterministic=True)
        
        # 2. The environment updates based on the AI's action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Safely extract resource cost if it exists in your info dict, otherwise default to 0
        cost = info.get('resource_cost', 0) 
        total_cost += cost
        
        # 3. Update the UI Dashboard
        growth_metric.metric("Plant Growth", f"{obs[0]:.3f}")
        moisture_metric.metric("Moisture", f"{obs[1]:.2f}")
        nutrient_metric.metric("Nutrients", f"{obs[2]:.2f}")
        temp_metric.metric("Temperature", f"{obs[3]:.2f}")
        light_metric.metric("Light", f"{obs[4]:.2f}")
        
        # 4. Log the hardware commands
        log_text += f"Day {day+1:02d} | Water: {action[0]:.2f} | Nutrients: {action[1]:.2f} | LED: {action[2]:.2f} | Daily Reward: {reward:.2f}\n"
        log_window.text_area("Live Log", log_text, height=250, label_visibility="collapsed")
        
        # Pause slightly so the professor can actually watch the numbers change
        time.sleep(0.1) 
        
        if terminated or truncated:
            st.success(f"🎉 Crop Cycle Complete! Total Resource Penalty: {total_cost:.2f}")
            break