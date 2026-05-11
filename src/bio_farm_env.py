import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BioFarmEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    # We add custom parameters here so Streamlit can change them!
    def __init__(self, base_growth_rate=0.02, penalty_multiplier=1.0, crop_cycle_days=60):
        super(BioFarmEnv, self).__init__()
        
        self.base_growth_rate = base_growth_rate
        self.penalty_multiplier = penalty_multiplier
        self.max_steps = crop_cycle_days
        
        # State: [Growth, Moisture, Nutrients, Temp, Light] 
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        # Actions: [Water, Nutrients, LED]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = np.array([0.01, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        water_action, nutrient_action, light_action = action
        growth, moisture, nutrients, temp, light = self.state
        
        # Physics of the environment
        moisture = np.clip(moisture - 0.05 + (water_action * 0.1), 0, 1) 
        nutrients = np.clip(nutrients - 0.02 + (nutrient_action * 0.05), 0, 1) 
        light = np.clip(0.2 + (light_action * 0.8), 0, 1) 
        
        # Growth logic uses the CUSTOM growth rate
        if (0.4 < moisture < 0.8 and 0.4 < nutrients < 0.8):
            growth_rate = self.base_growth_rate * light
        else:
            growth_rate = 0.002 # Stunted growth
            
        growth = np.clip(growth + growth_rate, 0, 1)
        self.state = np.array([growth, moisture, nutrients, temp, light], dtype=np.float32)
        
        # Base penalties
        base_penalty = (water_action * 0.05) + (nutrient_action * 0.1) + (light_action * 0.2)
        # Apply the CUSTOM penalty multiplier
        total_penalty = base_penalty * self.penalty_multiplier
        
        # Reward function
        reward = (growth_rate * 100) - total_penalty
        
        terminated = bool(growth >= 1.0) 
        truncated = bool(self.current_step >= self.max_steps) 
        
        return self.state, reward, terminated, truncated, {'growth': growth, 'resource_cost': total_penalty}