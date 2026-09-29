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