# 🌱 RL-Powered Bio-Farm Optimization 

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?logo=docker)](https://www.docker.com/)
[![CI/CD](https://github.com/YourUsername/RL-BioFarm-Optimization/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/YourUsername/RL-BioFarm-Optimization/actions)

## 📌 Problem Statement
Indoor farming is critical for future food security but often consumes unsustainable amounts of electricity and water. This project utilizes **Deep Reinforcement Learning (Proximal Policy Optimization & A2C)** to autonomously control a simulated Bio-Farm. The AI is penalized for excessive resource use and rewarded for plant growth, forcing it to discover the most efficient, sustainable agricultural strategies.

This project directly aligns with:
* **SDG 12:** Responsible Consumption and Production
* **SDG 13:** Climate Action

## 🏗️ System Architecture & MLOps Stack
This system is built using decoupled microservices and industry-standard MLOps practices.

* **Reinforcement Learning:** `Stable-Baselines3` & `Gymnasium` (Custom Continuous MDP Environment)
* **Experiment Tracking & Model Registry:** `MLflow` (Hyperparameter tuning logs)
* **Data Versioning:** `DVC` (Data Version Control for `farm_config.json` physics parameters)
* **Backend Inference API:** `FastAPI` (REST API with prediction logging)
* **Frontend Dashboard:** `Streamlit` (Interactive Simulation UI)
* **Orchestration:** `Docker` & `Docker Compose`
* **CI/CD:** `GitHub Actions` (Automated environment testing on push/PR)

---

## ⚙️ Setup & Installation

Follow these steps to run the containerized pipeline on your local machine.

### Prerequisites
* [Docker](https://www.docker.com/products/docker-desktop/) installed and running.
* [Git](https://git-scm.com/) installed.

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/RL-BioFarm-Optimization.git](https://github.com/YourUsername/RL-BioFarm-Optimization.git)
cd RL-BioFarm-Optimization
