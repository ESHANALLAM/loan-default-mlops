# Run INSIDE your empty project folder:  .\setup_project.ps1
# Creates every folder + file of the Loan Default MLOps project.
function W($p, $t) {
    $full = Join-Path (Get-Location) $p
    $d = Split-Path $full
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force $d | Out-Null }
    [IO.File]::WriteAllText($full, $t, (New-Object Text.UTF8Encoding $false))
    Write-Host "created $p" -ForegroundColor Green
}

W "requirements.txt" @'
pandas
numpy
scikit-learn
xgboost
joblib
mlflow
dvc
fastapi
uvicorn
pydantic
httpx
pytest
evidently==0.4.40
'@

W "requirements-api.txt" @'
pandas
numpy
scikit-learn
xgboost
joblib
fastapi
uvicorn
pydantic
'@

W ".gitignore" @'
.venv/
__pycache__/
*.pyc
mlruns/
mlflow.db
logs/
reports/
models/
data/loan_new.csv
'@

W ".dockerignore" @'
.venv
.git
mlruns
mlflow.db
data
reports
tests
'@

W "src/__init__.py" ""
W "tests/__init__.py" ""

W "data/make_data.py" @'
"""Generate a synthetic loan dataset. --drift makes a shifted 'new' dataset."""
import argparse, os
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=5000)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--drift", action="store_true")
p.add_argument("--out", default="data/loan.csv")
a = p.parse_args()

r = np.random.default_rng(a.seed)
n = a.n
age = r.integers(21, 66, n)
income = r.lognormal(11.0, 0.5, n).round(0)
loan = r.uniform(2000, 40000, n).round(0)
credit = np.clip(r.normal(680, 70, n), 300, 850).round()
emp = np.clip(r.gamma(2, 3, n), 0, 40).round(1)
dti = np.clip(r.normal(18, 8, n), 0, 50).round(1)
term = r.choice([36, 60], n)
opened = r.integers(1, 15, n)
home = r.choice(["RENT", "MORTGAGE", "OWN"], n, p=[0.45, 0.40, 0.15])
rate = np.clip(22 - (credit - 300) / 550 * 14 + r.normal(0, 1.5, n), 5, 30).round(2)

if a.drift:  # simulate a worse economy: lower income, higher rates, weaker credit
    income = (income * 0.75).round(0)
    rate = (rate + 3).round(2)
    credit = np.clip(credit - 30, 300, 850)
    dti = np.clip(dti + 6, 0, 50).round(1)

z = (-2.4 + 0.045 * dti + 0.09 * (rate - 12) - 0.006 * (credit - 680)
     + 1.2 * (loan / income) - 0.04 * emp - 0.3 * (home == "OWN") + r.normal(0, 0.5, n))
default = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)

df = pd.DataFrame(dict(age=age, annual_income=income, loan_amount=loan,
                       interest_rate=rate, loan_term_months=term, credit_score=credit,
                       employment_years=emp, dti=dti, open_accounts=opened,
                       home_ownership=home, default=default))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
df.to_csv(a.out, index=False)
print(f"saved {a.out}: {len(df)} rows, default rate = {df['default'].mean():.1%}")
'@

W "src/features.py" @'
import pandas as pd

TARGET = "default"
NUMERIC = ["age", "annual_income", "loan_amount", "interest_rate", "loan_term_months",
           "credit_score", "employment_years", "dti", "open_accounts"]
RAW = NUMERIC + ["home_ownership"]
HOME = {"RENT": 0, "MORTGAGE": 1, "OWN": 2}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[RAW].copy()
    X["home_ownership"] = X["home_ownership"].map(HOME).fillna(0).astype(int)
    X["loan_to_income"] = X["loan_amount"] / X["annual_income"].clip(lower=1)
    return X
'@

W "src/train.py" @'
"""Train XGBoost, track in MLflow, promote only if better (controlled update)."""
import argparse, hashlib, json, os
import joblib, mlflow, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from src.features import TARGET, prepare

p = argparse.ArgumentParser()
p.add_argument("--n-estimators", type=int, default=150)
p.add_argument("--max-depth", type=int, default=2)
p.add_argument("--lr", type=float, default=0.05)
p.add_argument("--run-name", default="xgb-run")
a = p.parse_args()

DATA, MODEL, METRICS = "data/loan.csv", "models/model.joblib", "models/metrics.json"
df = pd.read_csv(DATA)
X, y = prepare(df), df[TARGET]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

params = dict(n_estimators=a.n_estimators, max_depth=a.max_depth, learning_rate=a.lr,
              eval_metric="logloss", random_state=42)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("loan-default")

with mlflow.start_run(run_name=a.run_name):
    model = XGBClassifier(**params).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    m = dict(roc_auc=roc_auc_score(yte, proba), accuracy=accuracy_score(yte, pred),
             precision=precision_score(yte, pred, zero_division=0),
             recall=recall_score(yte, pred), f1=f1_score(yte, pred))
    mlflow.log_params(params)
    mlflow.log_param("data_rows", len(df))
    mlflow.log_param("data_md5", hashlib.md5(open(DATA, "rb").read()).hexdigest()[:10])
    mlflow.log_metrics(m)
    mlflow.xgboost.log_model(model, "model", registered_model_name="loan-default-model")

    # ---- controlled model update: promote only if ROC-AUC improves ----
    os.makedirs("models", exist_ok=True)
    best = json.load(open(METRICS))["roc_auc"] if os.path.exists(METRICS) else -1
    if m["roc_auc"] > best:
        joblib.dump(model, MODEL)
        json.dump(m, open(METRICS, "w"), indent=2)
        decision = f"PROMOTED (AUC {m['roc_auc']:.4f} > previous {best:.4f})"
    else:
        decision = f"REJECTED (AUC {m['roc_auc']:.4f} <= current {best:.4f})"
    mlflow.set_tag("decision", decision)

print("\n=== METRICS ===")
for k, v in m.items():
    print(f"{k:10s}: {v:.4f}")
print("=== DECISION ===\n" + decision)
imp = sorted(zip(X.columns, model.feature_importances_), key=lambda t: -t[1])[:5]
print("=== TOP FEATURES ===")
for k, v in imp:
    print(f"{k:18s} {v:.3f}")
'@

W "src/api.py" @'
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
'@

W "src/monitor.py" @'
"""Data drift report: training data (reference) vs new/production data (current)."""
import os, sys
import pandas as pd
from src.features import RAW

cur_path = sys.argv[1] if len(sys.argv) > 1 else "data/loan_new.csv"
ref = pd.read_csv("data/loan.csv")[RAW]
cur = pd.read_csv(cur_path)[RAW]
os.makedirs("reports", exist_ok=True)
out = "reports/drift_report.html"

try:  # evidently 0.4.x
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    rep = Report(metrics=[DataDriftPreset()])
    rep.run(reference_data=ref, current_data=cur)
    rep.save_html(out)
    res = rep.as_dict()["metrics"][0]["result"]
    share, n = res["share_of_drifted_columns"], res["number_of_drifted_columns"]
    print(f"Drifted columns: {n} of {len(RAW)} ({share:.0%})")
    print("DRIFT DETECTED -> retrain model" if share >= 0.3 else "No significant drift")
except ImportError:  # evidently 0.7+
    from evidently import Report
    from evidently.presets import DataDriftPreset
    Report([DataDriftPreset()]).run(cur, ref).save_html(out)
print("Report saved:", out)
'@

W "tests/test_api.py" @'
from fastapi.testclient import TestClient
from src.api import app

c = TestClient(app)
SAMPLE = dict(age=35, annual_income=55000, loan_amount=15000, interest_rate=14.5,
              loan_term_months=36, credit_score=640, employment_years=4,
              dti=22, open_accounts=6, home_ownership="RENT")


def test_health():
    assert c.get("/health").json()["status"] == "ok"


def test_predict():
    r = c.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    assert 0 <= r.json()["default_probability"] <= 1
'@

W "Dockerfile" @'
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src ./src
COPY models ./models
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
'@

W ".github/workflows/ci.yml" @'
name: CI
on: [push, pull_request]
jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -r requirements.txt
      - run: python data/make_data.py
      - run: python -m src.train
      - run: python -m pytest -q
      - run: docker build -t loan-default-api .
'@

Write-Host "`nProject files ready. Next: follow the steps in the chat." -ForegroundColor Cyan
