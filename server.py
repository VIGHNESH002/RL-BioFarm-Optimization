from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import json
import asyncio
import random

app = FastAPI()

# Read our frontend HTML file
@app.get("/")
async def get():
    with open("index.html", "r") as f:
        return HTMLResponse(f.read())

# The WebSocket that streams data to the browser
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Starting conditions
    growth = 0.01
    moisture = 0.5
    total_cost = 0.0
    day = 1
    
    try:
        while day <= 60:
            # Here, you would normally call: action = model.predict(obs)
            # For this demo, we simulate the AI's choices
            water_action = random.uniform(0.1, 0.5)
            led_action = random.uniform(0.2, 0.9)
            
            # Simulate Environment physics
            moisture = min(1.0, max(0.0, moisture - 0.05 + water_action))
            growth += 0.015 * led_action
            reward = (0.015 * 100) - (water_action * 0.05 + led_action * 0.2)
            total_cost += (water_action * 0.05 + led_action * 0.2)
            
            # Package the data to send to the crazy frontend
            data = {
                "day": day,
                "growth": min(1.0, growth),
                "moisture": moisture,
                "led_intensity": led_action,
                "reward": round(reward, 2),
                "total_cost": round(total_cost, 2)
            }
            
            # Send data to the browser
            await websocket.send_text(json.dumps(data))
            
            day += 1
            await asyncio.sleep(0.5) # Pause for half a second so we can see the animation!
            
    except Exception as e:
        print("Simulation ended or disconnected.")