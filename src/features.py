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