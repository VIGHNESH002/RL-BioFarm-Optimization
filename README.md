# Smart Bio-Farm Optimizer

This project implements a Deep Reinforcement Learning (RL) agent using the Proximal Policy Optimization (PPO) algorithm to optimize resource management in an indoor vertical Smart Bio-Farm. The agent dynamically controls water pumps, fertilizer dispensers, and LED grow lights to minimize energy/water costs while maximizing plant growth.

The project also incorporates professional MLOps concepts such as:
* model versioning
* reproducibility
* environment physics simulation
* Git-based version control
* and dashboard-based visualization.

## Sustainable Development Goals (SDG 12 & 13)

This project directly supports **SDG 12: Responsible Consumption and Production** and **SDG 13: Climate Action** by:
* **Maximizing Resource Efficiency:** The RL agent learns to intelligently apply exact amounts of water and nutrients, eliminating agricultural runoff and waste.
* **Minimizing Energy Consumption:** Reducing unnecessary LED light usage decreases the carbon footprint associated with indoor farming.
* **Enhancing Yield Optimization:** Balancing resource costs with biological growth ensures a reliable food supply with maximum economic and environmental efficiency.

## Environment Details

A custom gymnasium-compliant environment (src/environment.py) simulates a full 60-day crop cycle. It relies entirely on internal mathematical physics including:
* cumulative biological growth rates
* natural soil moisture evaporation
* nutrient soil depletion
* and LED thermal heat generation.

## Observation Space (Continuous)

The environment tracks 5 continuous sensor metrics (normalized between 0.0 and 1.0):
1. Plant Growth (0% to 100%)
2. Soil Moisture Level
3. Nutrient Concentration
4. Ambient Temperature
5. Light Intensity

## Action Space (Continuous)

| Action | Description | Output Range |
| :--- | :--- | :--- |
| 0 | Water Pump Intensity | 0.0 to 1.0 |
| 1 | Fertilizer Dispenser | 0.0 to 1.0 |
| 2 | LED Grow-Light Intensity | 0.0 to 1.0 |

## Reward Function

The reward is mathematically defined as:
reward = (growth_rate * 100) - resource_cost
(Where resource cost is a weighted sum of water, nutrient, and electricity usage).

The RL agent therefore learns to:
* minimize operational resource costs,
* reduce expensive electricity and water usage,
* maximize biological plant growth,
* and avoid Reward Hacking (starving the plant to save money).

## Project Structure

    biofarm_ai/
    ├── data/                     # Configuration and dynamic simulation parameters
    ├── models/                   # Saved versioned RL models
    ├── src/                      
    │   ├── environment.py        # Custom gymnasium environment physics
    │   └── train.py              # PPO training loop and evaluation script
    ├── visualization/            
    │   └── dashboard.py          # Streamlit dashboard for visualization/demo
    ├── requirements.txt          # Project dependencies
    └── README.md                 # Project documentation

## Technologies and Libraries Used

* Python 3.10+
* Stable-Baselines3 (PPO Algorithm)
* Gymnasium
* NumPy
* Streamlit
* Git & GitHub

## Installation & Setup

Clone the repository:
    git clone https://github.com/VIGHNESH002/biofarm_ai.git

Move into the project folder:
    cd biofarm_ai

Create a virtual environment and activate it:
    python -m venv venv
    .\venv\Scripts\activate

Install all required dependencies:
    pip install -r requirements.txt

## Usage Instructions

### 1. Train the RL Agent
Train the baseline PPO agent to generate the versioned AI brain:
    python src/train.py

Training generates a saved .zip model inside the models/ directory and outputs mean evaluation rewards.

### 2. Launch Streamlit Dashboard
The Streamlit dashboard provides:
* live simulation visualization,
* continuous sensor tracking,
* and a real-time hardware action log.

Run the dashboard:
    streamlit run visualization/dashboard.py

## Experiment Tracking & Versioning

The project utilizes Git tagging and file-based versioning to track model improvements. 
* Models are saved with specific version tags.
* Training scripts can be adjusted to track different timesteps and hyperparameter configurations across versions.

## Reproducibility

Experiments are reproducible via the src/train.py script. The PPO algorithm evaluates the policy dynamically, and deterministic prediction flags are used during the dashboard simulation to ensure consistent demonstration behavior.

## Monitoring Plan (Deployment Design)

If this AI were deployed in a real-world physical vertical farm, we would monitor the following:
* Average Resource Cost per Hour: Alert if water/electricity cost exceeds 20% above the training baseline.
* Moisture / Temperature Drift: If soil moisture stays below critical thresholds for 3 or more consecutive hours, hardware sensor recalibration or AI retraining may be required.
* Growth Velocity: Track the daily percentage of actual plant growth vs. simulated expected growth.
* Hardware Failure: Any command issued by the AI that is not executed by the IoT actuators triggers an immediate system alert.

## Results

The trained RL agent demonstrates:
* reduced operational cost,
* highly targeted resource application (water/nutrients),
* and automated 60-day cycle completion without human intervention.
Evaluations show the agent successfully balancing the high penalty of resources against the reward of plant growth.

## Limitations

Current limitations include:
* synthetic mathematical physics rather than real-world biological datasets,
* simplified thermal dynamics,
* no hardware wear-and-tear degradation modeling for the pumps and LEDs,
* and lack of real-time weather integration.

## Future Improvements

Potential future enhancements include:
* deploying standard Q-Learning or SAC algorithms for comparison against PPO,
* integrating real-world IoT sensor data streams,
* cloud deployment via Docker,
* and live predictive weather integration to offset LED usage with natural sunlight.

## Conclusion

This project successfully demonstrates the application of Deep Reinforcement Learning and MLOps concepts for smart agriculture optimization. The RL agent learns adaptive bio-management strategies capable of balancing physics and economics, supporting sustainable and efficient farming aligned with SDG 12 and SDG 13.
