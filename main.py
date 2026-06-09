# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load trained model
model = joblib.load("it_package_model.pkl")

class CareerInput(BaseModel):
    Experience: int
    CurrentPackage: float
    SkillsCount: int
    Certifications: int
    CodingLevel: int

@app.get("/")
def read_root():
    return {"message": "IT Career Upskilling Prediction API is live 🚀"}

@app.post("/predict")
def predict(data: CareerInput):
    input_data = np.array([[
        data.Experience,
        data.CurrentPackage,
        data.SkillsCount,
        data.Certifications,
        data.CodingLevel
    ]])

    prediction = model.predict(input_data)[0]

    return {
        "upskill_required": bool(prediction),
        "career_status": (
            "Strong Package Level 💼"
            if prediction == 0
            else "Upskilling Recommended 🚀"
        ),
        "suggestion": (
            "Focus on niche skills like System Design, Cloud Architecture, or Leadership"
            if prediction == 0
            else "Learn DevOps, Cloud and AI to unlock 10+ LPA opportunities"
        )
    }
