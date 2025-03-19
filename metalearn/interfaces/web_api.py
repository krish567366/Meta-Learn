from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
from typing import List

app = FastAPI(title="Metalearn API")

class AdaptationRequest(BaseModel):
    support_set: List[dict]
    adaptation_steps: int = 5

class PredictionRequest(BaseModel):
    inputs: List[float]
    context: dict = None

@app.post("/adapt")
async def adapt_model(request: AdaptationRequest):
    """Perform online adaptation with provided support set"""
    # Implementation would interface with the core meta-learner
    return {"status": "adaptation_successful"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make predictions using adapted model"""
    # Implementation would use the current model state
    return {"predictions": [0.5, 0.3]}  # Example output

@app.post("/deploy")
async def deploy_model(file: UploadFile = File(...)):
    """Deploy new base model"""
    contents = await file.read()
    # Implementation would load and verify the model
    return {"status": "model_deployed"}