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