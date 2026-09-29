# Run INSIDE your existing loan-default-mlops folder: .\add_frontend.ps1
# Adds a browser UI at http://localhost:8000/ so you have a real website to demo.
function W($p, $t) {
    $full = Join-Path (Get-Location) $p
    $d = Split-Path $full
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force $d | Out-Null }
    [IO.File]::WriteAllText($full, $t, (New-Object Text.UTF8Encoding $false))
    Write-Host "created $p" -ForegroundColor Green
}

W "static/index.html" @'
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Loan Default Risk Assessment</title>
<style>
  :root{--navy:#1E2A4A;--teal:#0F8B8D;--bg:#EEF3F7;--text:#2D3748;--low:#1B8A5A;--med:#C08A00;--high:#C0392B}
  *{box-sizing:border-box}
  body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--text);display:flex;justify-content:center;padding:32px 16px}
  .wrap{width:100%;max-width:920px}
  h1{color:var(--navy);margin:0 0 4px}
  p.sub{color:#5a677a;margin:0 0 24px}
  .card{background:#fff;border-radius:12px;padding:24px;box-shadow:0 2px 10px rgba(20,30,50,.08)}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px}
  label{display:block;font-size:13px;font-weight:600;color:var(--navy);margin-bottom:6px}
  input,select{width:100%;padding:9px 10px;border:1px solid #cdd6e0;border-radius:8px;font-size:14px}
  input:focus,select:focus{outline:2px solid var(--teal);border-color:var(--teal)}
  button{margin-top:20px;background:var(--navy);color:#fff;border:none;padding:12px 22px;border-radius:8px;font-size:15px;font-weight:600;cursor:pointer}
  button:hover{background:var(--teal)}
  button:disabled{opacity:.6;cursor:default}
  #result{margin-top:24px;display:none}
  .badge{display:inline-block;padding:4px 14px;border-radius:999px;color:#fff;font-weight:700;font-size:13px;letter-spacing:.03em}
  .low{background:var(--low)} .medium{background:var(--med)} .high{background:var(--high)}
  .stat{font-size:34px;font-weight:700;color:var(--navy)}
  .err{color:var(--high);margin-top:14px}
</style>
</head>
<body>
<div class="wrap">
  <h1>Loan Default Risk Assessment</h1>
  <p class="sub">Enter borrower details to get a live prediction from the deployed model.</p>
  <div class="card">
    <form id="f">
      <div class="grid">
        <div><label>Age</label><input name="age" type="number" value="35" required></div>
        <div><label>Annual income ($)</label><input name="annual_income" type="number" value="55000" required></div>
        <div><label>Loan amount ($)</label><input name="loan_amount" type="number" value="15000" required></div>
        <div><label>Interest rate (%)</label><input name="interest_rate" type="number" step="0.1" value="14.5" required></div>
        <div><label>Loan term (months)</label>
          <select name="loan_term_months"><option value="36">36</option><option value="60">60</option></select>
        </div>
        <div><label>Credit score</label><input name="credit_score" type="number" value="640" required></div>
        <div><label>Employment (years)</label><input name="employment_years" type="number" step="0.1" value="4" required></div>
        <div><label>Debt-to-income (%)</label><input name="dti" type="number" step="0.1" value="22" required></div>
        <div><label>Open accounts</label><input name="open_accounts" type="number" value="6" required></div>
        <div><label>Home ownership</label>
          <select name="home_ownership"><option>RENT</option><option>MORTGAGE</option><option>OWN</option></select>
        </div>
      </div>
      <button type="submit" id="btn">Assess risk</button>
    </form>
    <div id="result">
      <div class="stat" id="prob">-</div>
      <div>Default probability</div>
      <p style="margin-top:14px"><span class="badge" id="badge">-</span>
         &nbsp; Decision: <strong id="decision">-</strong></p>
    </div>
    <div class="err" id="err"></div>
  </div>
</div>
<script>
const f = document.getElementById('f');
f.addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('btn');
  const err = document.getElementById('err');
  const result = document.getElementById('result');
  err.textContent = ''; result.style.display = 'none'; btn.disabled = true; btn.textContent = 'Scoring...';
  const data = Object.fromEntries(new FormData(f).entries());
  for (const k of ['age','annual_income','loan_amount','interest_rate','loan_term_months',
                   'credit_score','employment_years','dti','open_accounts']) data[k] = Number(data[k]);
  try {
    const r = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)});
    if (!r.ok) throw new Error('Server returned ' + r.status);
    const j = await r.json();
    document.getElementById('prob').textContent = (j.default_probability*100).toFixed(1) + '%';
    const b = document.getElementById('badge');
    b.textContent = j.risk_level; b.className = 'badge ' + j.risk_level.toLowerCase();
    document.getElementById('decision').textContent = j.decision;
    result.style.display = 'block';
  } catch (ex) {
    err.textContent = 'Could not reach the API: ' + ex.message;
  } finally {
    btn.disabled = false; btn.textContent = 'Assess risk';
  }
});
</script>
</body>
</html>
'@

W "src/api.py" @'
import csv, datetime, os
from pathlib import Path
import joblib, pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from src.features import RAW, prepare

app = FastAPI(title="Loan Default Prediction API", version="1.0")
model = joblib.load("models/model.joblib")
LOG = "logs/predictions.csv"


@app.get("/", response_class=HTMLResponse)
def home():
    return Path("static/index.html").read_text(encoding="utf-8")


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
    os.makedirs("logs", exist_ok=True)
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(RAW + ["probability", "timestamp"])
        w.writerow([row[c] for c in RAW] + [round(prob, 4), datetime.datetime.now().isoformat()])
    return {"default_probability": round(prob, 4), "risk_level": level, "decision": action}
'@

Write-Host "`nFrontend added. Run: uvicorn src.api:app --reload   then open http://localhost:8000" -ForegroundColor Cyan
