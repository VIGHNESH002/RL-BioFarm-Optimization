from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from stable_baselines3 import PPO, A2C
import logging

# Logging for Monitoring (Rubric 2)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Bio-Farm Inference API")

try:
    # Try loading the best model saved from our tuning script
    model = PPO.load("best_production_model")
    logger.info("✅ Best Production Model loaded.")
except:
    model = None
    logger.warning("⚠️ No model found! Run train_agent.py first.")

class SensorData(BaseModel):
    growth: float
    moisture: float
    nutrients: float
    temp: float
    light: float

@app.get("/")
def health_check():
    return {"status": "API is live", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: SensorData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    obs = [data.growth, data.moisture, data.nutrients, data.temp, data.light]
    action, _ = model.predict(obs, deterministic=True)
    
    logger.info(f"Sensors: {obs} | Predicted Action: {action.tolist()}")
    
    return {
        "water_action": float(action[0]),
        "nutrient_action": float(action[1]),
        "led_action": float(action[2])
    }