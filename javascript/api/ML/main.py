# ==================================================
# IMPORTS
# ==================================================

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

import pickle


# ==================================================
# FASTAPI
# ==================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[3]

MODELS_DIR = BASE_DIR / "models"

MODEL_VERSION = "v1"


# ==================================================
# LOAD MODEL
# ==================================================

with open(MODELS_DIR / f"{MODEL_VERSION}_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(MODELS_DIR / f"{MODEL_VERSION}_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(MODELS_DIR / f"{MODEL_VERSION}_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


# ==================================================
# REQUEST BODY JSON
# ==================================================

class ClaimRequest(BaseModel):
    user_claim: str


# ==================================================
# HOME
# ==================================================

@app.get("/")
def home():
    return {"message": "API OK"}


# ==================================================
# PREDICT
# ==================================================

@app.post("/predict")
def predict(data: ClaimRequest):

    # texte utilisateur
    text = data.user_claim.lower()

    # vectorisation
    X = vectorizer.transform([text])

    # prediction
    pred = model.predict(X)

    # decode label
    label = label_encoder.inverse_transform(pred)

    return {
        "prediction": label[0]
    }