# Run INSIDE your existing loan-default-mlops folder: .\upgrade_dashboard.ps1
# Upgrades the site to a full dashboard: gauge chart, model metrics, feature
# importance chart, and a recent-predictions table. Also adds render.yaml for
# free public hosting.
function W($p, $t) {
    $full = Join-Path (Get-Location) $p
    $d = Split-Path $full
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force $d | Out-Null }
    [IO.File]::WriteAllText($full, $t, (New-Object Text.UTF8Encoding $false))
    Write-Host "updated $p" -ForegroundColor Green
}

W "src/train.py" @'
"""Train XGBoost, track in MLflow, promote only if better (controlled update)."""
import argparse, datetime, hashlib, json, os
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
    imp = sorted(zip(X.columns, model.feature_importances_.tolist()), key=lambda t: -t[1])[:6]
    if m["roc_auc"] > best:
        joblib.dump(model, MODEL)
        meta = dict(m, feature_importance=imp, params=params, data_rows=len(df),
                    trained_at=datetime.datetime.now().isoformat(timespec="seconds"))
        json.dump(meta, open(METRICS, "w"), indent=2)
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
import csv, datetime, json, os
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


@app.get("/metrics")
def metrics():
    if not os.path.exists("models/metrics.json"):
        return {}
    return json.load(open("models/metrics.json"))


@app.get("/recent")
def recent(n: int = 8):
    if not os.path.exists(LOG):
        return []
    df = pd.read_csv(LOG).tail(n).iloc[::-1]
    return df.to_dict(orient="records")


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

W "static/index.html" @'
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Loan Default Risk Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  :root{
    --navy:#101b34; --navy2:#1e2f57; --teal:#12b3a8; --teal-d:#0d8a82;
    --violet:#7c6cf0; --bg:#f3f6fb; --card:#ffffff; --text:#1c2434; --muted:#6b7690;
    --low:#1fa06a; --med:#e0a512; --high:#e5484d; --border:#e4e9f2;
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:'Inter',Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--text)}
  header{background:linear-gradient(120deg,var(--navy) 0%,var(--navy2) 55%,#2c3f78 100%);color:#fff;padding:36px 6vw 46px}
  header .row{display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px}
  header h1{margin:0 0 6px;font-size:30px;font-weight:800;letter-spacing:-.02em}
  header p{margin:0;color:#b8c4e6;font-size:15px;max-width:560px}
  .pill{display:inline-flex;align-items:center;gap:8px;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.18);
        padding:7px 14px;border-radius:999px;font-size:13px;font-weight:600}
  .pill .dot{width:8px;height:8px;border-radius:50%;background:#3ee08a;box-shadow:0 0 0 4px rgba(62,224,138,.25)}
  main{max-width:1180px;margin:-26px auto 60px;padding:0 6vw;display:grid;gap:20px}
  .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
  .stat-card{background:var(--card);border-radius:14px;padding:18px 20px;box-shadow:0 6px 22px rgba(16,27,52,.08);border:1px solid var(--border)}
  .stat-card .lbl{font-size:12.5px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
  .stat-card .val{font-size:28px;font-weight:800;color:var(--navy);margin-top:4px}
  .grid2{display:grid;grid-template-columns:1.15fr .85fr;gap:20px}
  .card{background:var(--card);border-radius:16px;padding:26px;box-shadow:0 6px 22px rgba(16,27,52,.07);border:1px solid var(--border)}
  .card h2{margin:0 0 4px;font-size:18px;font-weight:800;color:var(--navy)}
  .card .sub{margin:0 0 18px;color:var(--muted);font-size:13.5px}
  .fgrid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  label{display:block;font-size:12.5px;font-weight:700;color:var(--navy2);margin-bottom:6px}
  input,select{width:100%;padding:10px 12px;border:1.5px solid var(--border);border-radius:9px;font-size:14px;font-family:inherit;background:#fbfcfe;transition:.15s}
  input:focus,select:focus{outline:none;border-color:var(--teal);box-shadow:0 0 0 3px rgba(18,179,168,.15)}
  button{margin-top:20px;width:100%;background:linear-gradient(120deg,var(--teal),var(--teal-d));color:#fff;border:none;
         padding:13px;border-radius:10px;font-size:15px;font-weight:700;cursor:pointer;letter-spacing:.01em;transition:.15s}
  button:hover{filter:brightness(1.08);transform:translateY(-1px)}
  button:disabled{opacity:.6;cursor:default;transform:none}
  .resultwrap{display:flex;flex-direction:column;align-items:center;text-align:center;justify-content:center;min-height:340px}
  .gaugebox{position:relative;width:200px;height:200px}
  .gaugebox canvas{width:100%!important;height:100%!important}
  .gaugenum{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
  .gaugenum .n{font-size:34px;font-weight:800;color:var(--navy)}
  .gaugenum .l{font-size:11.5px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.05em}
  .badge{display:inline-flex;align-items:center;gap:6px;padding:6px 16px;border-radius:999px;color:#fff;font-weight:800;
         font-size:12.5px;letter-spacing:.04em;margin-top:18px}
  .low{background:var(--low)} .medium{background:var(--med)} .high{background:var(--high)}
  .decision{margin-top:10px;font-size:15px;color:var(--navy2)}
  .decision b{color:var(--navy)}
  .placeholder{color:var(--muted);font-size:14px}
  .err{color:var(--high);margin-top:14px;font-size:13.5px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th{text-align:left;color:var(--muted);font-weight:700;text-transform:uppercase;font-size:10.5px;letter-spacing:.04em;
     padding:8px 10px;border-bottom:1.5px solid var(--border)}
  td{padding:9px 10px;border-bottom:1px solid var(--border);color:var(--navy2)}
  tr:last-child td{border-bottom:none}
  .rl{display:inline-block;padding:2px 9px;border-radius:999px;color:#fff;font-size:10.5px;font-weight:700}
  .empty-row td{color:var(--muted);text-align:center;padding:22px}
  @media(max-width:880px){.grid2{grid-template-columns:1fr}.fgrid{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>
<header>
  <div class="row">
    <div>
      <h1>Loan Default Risk Dashboard</h1>
      <p>Live borrower scoring backed by an XGBoost model tracked in MLflow and served through FastAPI.</p>
    </div>
    <span class="pill"><span class="dot"></span> Model live</span>
  </div>
</header>

<main>
  <div class="stats" id="statCards"></div>

  <div class="grid2">
    <div class="card">
      <h2>New Application</h2>
      <p class="sub">Enter borrower details and score in real time.</p>
      <form id="f">
        <div class="fgrid">
          <div><label>Age</label><input name="age" type="number" value="35" required></div>
          <div><label>Annual income ($)</label><input name="annual_income" type="number" value="55000" required></div>
          <div><label>Loan amount ($)</label><input name="loan_amount" type="number" value="15000" required></div>
          <div><label>Interest rate (%)</label><input name="interest_rate" type="number" step="0.1" value="14.5" required></div>
          <div><label>Loan term (months)</label>
            <select name="loan_term_months"><option value="36">36</option><option value="60">60</option></select>
          </div>
          <div><label>Credit score</label><input name="credit_score" type="number" value="640" required></div>
          <div><label>Employment (yrs)</label><input name="employment_years" type="number" step="0.1" value="4" required></div>
          <div><label>Debt-to-income (%)</label><input name="dti" type="number" step="0.1" value="22" required></div>
          <div><label>Open accounts</label><input name="open_accounts" type="number" value="6" required></div>
          <div><label>Home ownership</label>
            <select name="home_ownership"><option>RENT</option><option>MORTGAGE</option><option>OWN</option></select>
          </div>
        </div>
        <button type="submit" id="btn">Score application</button>
        <div class="err" id="err"></div>
      </form>
    </div>

    <div class="card">
      <h2>Prediction</h2>
      <p class="sub">Default probability from the current model.</p>
      <div class="resultwrap" id="resultwrap">
        <span class="placeholder">Submit the form to see a live prediction.</span>
      </div>
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <h2>Top Risk Drivers</h2>
      <p class="sub">Feature importance from the current registered model.</p>
      <canvas id="impChart" height="200"></canvas>
    </div>
    <div class="card">
      <h2>Recent Predictions</h2>
      <p class="sub">Latest requests served by the API.</p>
      <table id="recentTable">
        <thead><tr><th>Credit</th><th>DTI</th><th>Prob.</th><th>Risk</th></tr></thead>
        <tbody><tr class="empty-row"><td colspan="4">No predictions yet</td></tr></tbody>
      </table>
    </div>
  </div>
</main>

<script>
const RISK_COLOR = {LOW:'#1fa06a', MEDIUM:'#e0a512', HIGH:'#e5484d'};
let gauge, impChart;

function statCard(lbl, val){
  return `<div class="stat-card"><div class="lbl">${lbl}</div><div class="val">${val}</div></div>`;
}

async function loadMetrics(){
  try{
    const m = await (await fetch('/metrics')).json();
    const box = document.getElementById('statCards');
    if (!m || !m.roc_auc){ box.innerHTML = statCard('Status','No model yet'); return; }
    box.innerHTML =
      statCard('ROC-AUC', m.roc_auc.toFixed(3)) +
      statCard('Accuracy', (m.accuracy*100).toFixed(1)+'%') +
      statCard('Precision', (m.precision*100).toFixed(1)+'%') +
      statCard('Recall', (m.recall*100).toFixed(1)+'%');

    const labels = (m.feature_importance||[]).map(f=>f[0]);
    const values = (m.feature_importance||[]).map(f=>f[1]);
    if (impChart) impChart.destroy();
    impChart = new Chart(document.getElementById('impChart'), {
      type:'bar',
      data:{labels, datasets:[{data:values, backgroundColor:'#12b3a8', borderRadius:6, maxBarThickness:26}]},
      options:{indexAxis:'y', plugins:{legend:{display:false}},
        scales:{x:{grid:{color:'#eef1f7'}}, y:{grid:{display:false}}}}
    });
  }catch(e){ console.error(e); }
}

function riskBadge(level){ return `<span class="rl" style="background:${RISK_COLOR[level]}">${level}</span>`; }

async function loadRecent(){
  try{
    const rows = await (await fetch('/recent?n=8')).json();
    const body = document.querySelector('#recentTable tbody');
    if (!rows.length){ body.innerHTML = '<tr class="empty-row"><td colspan="4">No predictions yet</td></tr>'; return; }
    body.innerHTML = rows.map(r => `<tr>
        <td>${r.credit_score}</td><td>${r.dti}%</td>
        <td>${(r.probability*100).toFixed(1)}%</td>
        <td>${riskBadge(r.probability<0.2?'LOW':r.probability<0.5?'MEDIUM':'HIGH')}</td>
      </tr>`).join('');
  }catch(e){ console.error(e); }
}

function drawGauge(prob, level){
  const wrap = document.getElementById('resultwrap');
  wrap.innerHTML = `
    <div class="gaugebox">
      <canvas id="gaugeCanvas"></canvas>
      <div class="gaugenum"><div class="n">${(prob*100).toFixed(1)}%</div><div class="l">default prob.</div></div>
    </div>
    <span class="badge ${level.toLowerCase()}">${level} RISK</span>
    <div class="decision" id="decisionLine"></div>`;
  if (gauge) gauge.destroy();
  gauge = new Chart(document.getElementById('gaugeCanvas'), {
    type:'doughnut',
    data:{datasets:[{data:[prob, 1-prob], backgroundColor:[RISK_COLOR[level], '#eef1f7'], borderWidth:0}]},
    options:{cutout:'78%', circumference:360, rotation:0,
      plugins:{legend:{display:false}, tooltip:{enabled:false}}, animation:{animateRotate:true}}
  });
}

const f = document.getElementById('f');
f.addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('btn');
  const err = document.getElementById('err');
  err.textContent = ''; btn.disabled = true; btn.textContent = 'Scoring...';
  const data = Object.fromEntries(new FormData(f).entries());
  for (const k of ['age','annual_income','loan_amount','interest_rate','loan_term_months',
                   'credit_score','employment_years','dti','open_accounts']) data[k] = Number(data[k]);
  try{
    const r = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)});
    if (!r.ok) throw new Error('Server returned ' + r.status);
    const j = await r.json();
    drawGauge(j.default_probability, j.risk_level);
    document.getElementById('decisionLine').innerHTML = 'Decision: <b>' + j.decision + '</b>';
    loadRecent();
  }catch(ex){
    err.textContent = 'Could not reach the API: ' + ex.message;
  }finally{
    btn.disabled = false; btn.textContent = 'Score application';
  }
});

loadMetrics();
loadRecent();
</script>
</body>
</html>
'@

W "render.yaml" @'
services:
  - type: web
    name: loan-default-api
    env: docker
    plan: free
    dockerfilePath: ./Dockerfile
'@

Write-Host "`nDashboard upgraded. Retrain once so the new metrics file is written:" -ForegroundColor Cyan
Write-Host "  Remove-Item models\metrics.json -ErrorAction SilentlyContinue"
Write-Host "  python -m src.train"
Write-Host "Then run:  uvicorn src.api:app --reload   and open http://localhost:8000" -ForegroundColor Cyan
