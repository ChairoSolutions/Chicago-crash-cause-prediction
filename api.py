from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI(title="Chicago Crash Cause Prediction API v1.0")

deployment_dir = Path("deployment_artifacts")

model = joblib.load(deployment_dir / "model.joblib")

if hasattr(model, "named_steps") and "model" in model.named_steps:
    model.named_steps["model"].n_jobs = 1

class CrashInput(BaseModel):
    WEATHER_CONDITION: str
    LIGHTING_CONDITION: str 
    ROADWAY_SURFACE_COND: str
    TRAFFIC_CONTROL_DEVICE: str
    TRAFFICWAY_TYPE: str
    ALIGNMENT: str
    DEVICE_CONDITION: str
    dominant_maneuver: str
    dominant_sex: str
    avg_age: float
    num_vehicle_types: int
    CRASH_HOUR: int
    CRASH_DAY_OF_WEEK: int
    CRASH_MONTH: int    
    num_people: int
    POSTED_SPEED_LIMIT: int

@app.get("/")
def home():
    return {"message": "Welcome to the Chicago Crash Cause Prediction API v1.0"}

@app.post("/predict")
def predict(data: CrashInput):
    input_data = pd.DataFrame([data.dict()])

    prediction = model.predict(input_data)[0]

    response = {
        "predicted_crash_cause": str(prediction)
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        classes = model.classes_

        top_probabilities = (
            pd.DataFrame({
                "cause": classes,
                "probability": probabilities,
            })
            .sort_values("probability", ascending=False)
            .head(5)
        )

        response["top_probabilities"] = top_probabilities.to_dict(orient="records")

    return response
