# 🌱 Autonomous Bio-Farm Optimization Engine

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow" />
  <img src="https://img.shields.io/badge/DVC-945DD6?style=for-the-badge&logo=dvc&logoColor=white" alt="DVC" />
</div>

<br>

> **An end-to-end MLOps pipeline leveraging Deep Reinforcement Learning to optimize resource consumption in indoor vertical farming.** This project aligns with **SDG 12 (Responsible Consumption and Production)** and **SDG 13 (Climate Action)** by forcing an AI agent to discover the mathematical "sweet spot" between maximum plant growth and minimum water/energy usage.

---

## 🏗️ System Architecture & Workflow

This project abandons the monolithic script approach in favor of decoupled, production-ready microservices:

1. **The Brain (Algorithm):** `Stable-Baselines3` (PPO/A2C) operating within a custom `Gymnasium` Markov Decision Process (MDP).
2. **The Backend (Inference):** `FastAPI` serves the trained model via a RESTful API, handling real-time sensor data and logging predictions.
3. **The Frontend (UI):** `Streamlit` provides an interactive simulation dashboard for users to tweak physics parameters and visualize the AI's strategy.
4. **The Engine (Orchestration):** `Docker Compose` containerizes both services, ensuring they run identically on any machine.

---

## 🔐 Strict Versioning & MLOps Strategy (CO1, CO2, CO3)

In Reinforcement Learning, data is generated dynamically. To ensure 100% reproducibility, this project implements a strict, multi-tiered versioning architecture:

* 📦 **Environment Versioning (`Docker` & `requirements.txt`):** Guarantees that the underlying Python libraries and OS-level dependencies never shift unpredictably.
* 💾 **Data & Physics Versioning (`DVC`):** Because we do not use static CSVs, the physical rules of the simulation (growth rates, penalty multipliers) are extracted into `farm_config.json`. **Data Version Control (DVC)** tracks changes to this file independently of the codebase.
* 🧠 **Model Versioning & Registry (`MLflow`):** An automated hyperparameter tuning script compares PPO vs. A2C algorithms. MLflow tracks all metrics (Mean Reward, Resource Penalty) and automatically registers the highest-performing model artifact for production use.
* 🌿 **Code Versioning (`Git` & `GitHub Actions`):** Enforces a professional `dev` ➡️ `main` branching strategy utilizing Pull Requests. A CI/CD pipeline triggers automatically on every push to test the environment integrity.

---

## 🗂️ Repository Structure

```text
RL-BioFarm-Optimization/
│
├── .github/workflows/    # CI/CD pipelines (GitHub Actions)
├── .dvc/                 # Data Version Control configuration
├── mlruns/               # Local MLflow tracking database
│
├── api.py                # FastAPI Backend Service
├── app.py                # Streamlit Frontend Dashboard
├── bio_farm_env.py       # Custom Gymnasium Environment definition
├── train_agent.py        # MLflow hyperparameter tuning & training script
│
├── farm_config.json      # Dynamic physics parameters (Tracked via DVC)
├── farm_config.json.dvc  # DVC pointer file
│
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-container orchestration
├── requirements.txt      # Locked environment dependencies
└── README.md             # Project documentation
