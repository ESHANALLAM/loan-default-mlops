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