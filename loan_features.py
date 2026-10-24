"""Custom feature engineering for the loan-default project.

Everything that must be LEARNED (income cap, income percentiles, median income per
employment length) is learned in fit() from training data, and only APPLIED in transform().
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

MAX_EMP_YEARS = 50


class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, income_cap_quantile=0.995, growth_rate=0.03):
        self.income_cap_quantile = income_cap_quantile
        self.growth_rate = growth_rate

    def fit(self, X, y=None):
        income = X["person_income"]
        self.income_cap_ = float(income.quantile(self.income_cap_quantile))
        capped = income.clip(upper=self.income_cap_)

        # 1001 points that describe the income distribution of the TRAINING data
        self.income_grid_ = np.sort(capped.quantile(np.linspace(0, 1, 1001)).to_numpy())

        # median income for each employment length (in whole years)
        emp = X["person_emp_length"].clip(0, MAX_EMP_YEARS).round()
        self.income_by_emp_ = capped.groupby(emp).median().to_dict()
        self.overall_median_income_ = float(capped.median())
        return self

    def transform(self, X):
        df = X.copy()
        df = df.drop(columns=["person_age"], errors="ignore")      # redundant with credit history length

        # cap impossible / extreme values (the caps were learned in fit)
        df["person_emp_length"] = df["person_emp_length"].clip(0, MAX_EMP_YEARS)
        df["person_income"] = df["person_income"].clip(upper=self.income_cap_)

        # 1. where does this income stand compared to the training applicants?
        position = np.searchsorted(self.income_grid_, df["person_income"].to_numpy(), side="right")
        df["income_percentile"] = np.clip(position / len(self.income_grid_), 0, 1)

        # 2. income per year of credit history
        df["income_to_credit_history"] = df["person_income"] / df["cb_person_cred_hist_length"].clip(lower=1)

        # 3. income adjusted for years at work (assumed yearly growth)
        df["estimated_annual_income"] = df["person_income"] / (1 + self.growth_rate * df["person_emp_length"])

        # 4. typical income of people with the same employment length (from training data)
        emp_key = df["person_emp_length"].round()
        df["median_income_emp_group"] = emp_key.map(self.income_by_emp_).fillna(self.overall_median_income_)
        return df
