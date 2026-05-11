# 🌱 Autonomous Bio-Farm Optimization Engine

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Stable_Baselines3-20124d?style=for-the-badge" alt="Stable Baselines3" />
  <img src="https://img.shields.io/badge/Gymnasium-000000?style=for-the-badge" alt="Gymnasium" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Streamlit" />
</div>

<br>

> **An interactive AI system leveraging Deep Reinforcement Learning to optimize resource consumption in indoor vertical farming.** This project aligns with **SDG 12 (Responsible Consumption)** and **SDG 13 (Climate Action)** by forcing an AI agent to discover the mathematical "sweet spot" between maximum plant growth and minimum water/energy usage.

---

## 🏗️ System Architecture 

1. **The Environment:** A custom `Gymnasium` Markov Decision Process (MDP). The physical rules of the farm (growth rates, resource penalties) are mathematically defined here.
2. **The Brain:** We utilize **Proximal Policy Optimization (PPO)**. The agent features an Actor-Critic neural network architecture to process continuous sensor data and output continuous hardware controls.
3. **The Dashboard:** A `Streamlit` front-end provides an interactive simulation UI. Users can alter the physics of the environment (e.g., simulate a drought by increasing water costs) and watch the AI dynamically adapt its strategy.

---

## 🗂️ Version Control & Workflow
This repository follows standard software engineering practices:
* **Code Versioning:** Managed via `Git` and hosted on GitHub.
* **Environment Versioning:** All dependencies are locked within `requirements.txt` to ensure cross-platform reproducibility.
* **Branching Strategy:** Developed using an isolated `dev` branch and merged into `main` via Pull Requests.

---

## 🚀 Setup & Installation

Follow these steps to run the Reinforcement Learning simulation on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/RL-BioFarm-Optimization.git](https://github.com/VIGHNESH002/RL-BioFarm-Optimization.git)
cd RL-BioFarm-Optimization
