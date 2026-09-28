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