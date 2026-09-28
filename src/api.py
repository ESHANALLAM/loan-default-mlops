import csv, datetime, os
import joblib, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.features import RAW, prepare

app = FastAPI(title="Loan Default Prediction API", version="1.0")
model = joblib.load("models/model.joblib")
LOG = "logs/predictions.csv"


class Loan(BaseModel):
    age: int = Field(ge=18, le=100)
    annual_income: float = Field(gt=0)
    loan_amount: float = Field(gt=0)
    interest_rate: float
    loan_term_months: int
    credit_score: int = Field(ge=300, le=850)
    employment_years: float = Field(ge=0)
    dti: float = Field(ge=0)
    open_accounts: int = Field(ge=0)
    home_ownership: str = "RENT"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(loan: Loan):
    row = loan.model_dump()
    prob = float(model.predict_proba(prepare(pd.DataFrame([row])))[0, 1])
    level, action = (("LOW", "APPROVE") if prob < 0.2 else
                     ("MEDIUM", "MANUAL REVIEW") if prob < 0.5 else ("HIGH", "REJECT"))
    os.makedirs("logs", exist_ok=True)  # log inputs so Evidently can monitor them
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(RAW + ["probability", "timestamp"])
        w.writerow([row[c] for c in RAW] + [round(prob, 4), datetime.datetime.now().isoformat()])
    return {"default_probability": round(prob, 4), "risk_level": level, "decision": action}