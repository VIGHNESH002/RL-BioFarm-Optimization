import streamlit as st
import numpy as np
import time
import os
from stable_baselines3 import PPO
from bio_farm_env import BioFarmEnv

st.set_page_config(page_title="RL Bio-Farm Optimizer", layout="wide")

st.title("🌱 Adjustable RL Bio-System Dashboard")
st.markdown("Modify the physical parameters of the farm and observe how the AI adapts.")

# --- SIDEBAR: Custom Parameters ---
st.sidebar.header("⚙️ Environment Parameters")
st.sidebar.markdown("Change these to alter the physics of the farm.")

custom_growth_rate = st.sidebar.slider("Plant Growth Rate", min_value=0.01, max_value=0.05, value=0.02, step=0.01, help="Higher means the plant grows faster.")
custom_penalty_mult = st.sidebar.slider("Resource Cost Multiplier", min_value=0.5, max_value=3.0, value=1.0, step=0.5, help="Increase this to make water/energy more expensive, forcing the AI to be frugal.")
custom_days = st.sidebar.slider("Max Crop Cycle (Days)", min_value=30, max_value=90, value=60, step=10)

st.sidebar.markdown("---")

# --- SIDEBAR: Training Controls ---
st.sidebar.header("🧠 Agent Training")
if st.sidebar.button("Train Agent on New Parameters"):
    with st.spinner("Training RL Model (50,000 timesteps)..."):
        # Pass the UI variables into the Environment!
        env = BioFarmEnv(base_growth_rate=custom_growth_rate, penalty_multiplier=custom_penalty_mult, crop_cycle_days=custom_days)
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=50000)
        model.save("ppo_custom_model")
        st.sidebar.success("✅ Training Complete!")

model_exists = os.path.exists("ppo_custom_model.zip")
if model_exists:
    st.sidebar.success("✅ Model Ready for Simulation")
else:
    st.sidebar.error("❌ No model found. Train agent first.")
    st.stop()

# --- MAIN DASHBOARD ---
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.subheader("Sensors")
    moisture_bar = st.progress(0, text="Soil Moisture")
    nutrient_bar = st.progress(0, text="Nutrient Level")

with col2:
    st.subheader("Visual Environment")
    farm_display = st.empty()
    action_log = st.empty()

with col3:
    st.subheader("Metrics")
    day_metric = st.metric("Current Day", f"0/{custom_days}")
    growth_metric = st.metric("Plant Growth", "0%")
    cost_metric = st.metric("Total Penalty", "0.00")
    reward_feedback = st.empty()

# --- SIMULATION LOOP ---
if st.button("🚀 Run Simulation with Current Parameters"):
    # Pass the UI variables into the simulation environment!
    env = BioFarmEnv(base_growth_rate=custom_growth_rate, penalty_multiplier=custom_penalty_mult, crop_cycle_days=custom_days)
    model = PPO.load("ppo_custom_model")
    obs, _ = env.reset()
    total_cost = 0
    
    for day in range(1, custom_days + 1):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        growth, moisture, nutrients, temp, light = obs
        daily_cost = info['resource_cost']
        total_cost += daily_cost
        
        # Update Dashboard Visuals
        with farm_display:
            if growth < 0.33:
                st.markdown("<h1 style='text-align:center; font-size:100px;'>🌱</h1>", unsafe_allow_html=True)
            elif growth < 0.66:
                st.markdown("<h1 style='text-align:center; font-size:100px;'>🌿</h1>", unsafe_allow_html=True)
            else:
                st.markdown("<h1 style='text-align:center; font-size:100px;'>🌳</h1>", unsafe_allow_html=True)
                
        moisture_bar.progress(float(moisture), text=f"Moisture: {moisture:.2f}")
        nutrient_bar.progress(float(nutrients), text=f"Nutrients: {nutrients:.2f}")
        day_metric.metric("Day", f"{day}/{custom_days}")
        growth_metric.metric("Growth", f"{growth:.1%}")
        cost_metric.metric("Penalty", f"{total_cost:.2f}", delta=f"{daily_cost:.2f}", delta_color="inverse")
        action_log.code(f"Actions: Water={action[0]:.2f} | Nut={action[1]:.2f} | LED={action[2]:.2f}")
        
        if reward > 0:
            reward_feedback.markdown(f"<p style='color:green; font-weight:bold;'>Reward: +{reward:.2f}</p>", unsafe_allow_html=True)
        else:
            reward_feedback.markdown(f"<p style='color:red; font-weight:bold;'>Penalty: {reward:.2f}</p>", unsafe_allow_html=True)

        time.sleep(0.1)
        
        if terminated or truncated:
            st.balloons()
            st.success(f"Cycle Completed! Final Resource Penalty: {total_cost:.2f}")
            break