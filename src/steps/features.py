"""Etapa 4 (features). Faixas fixas (não aprendem nada dos dados)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BmiAgeBins(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Categorias da OMS para IMC
        X["BMI_cat"] = pd.cut(X["BMI"], [-np.inf, 18.5, 25, 30, np.inf], labels=False)
        X["Age_bin"] = pd.cut(X["Age"], [-np.inf, 30, 40, 50, np.inf], labels=False)
        return X


class Ratios(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Insulin_Glucose"] = X["Insulin"] / X["Glucose"].replace(0, np.nan)
        X["Glucose_BMI"] = X["Glucose"] * X["BMI"]
        X["Age_Pregnancies"] = X["Pregnancies"] / X["Age"]
        return X


class Homa(BaseEstimator, TransformerMixin):
    """HOMA-IR ≈ Glucose (mg/dL) x Insulin (µU/mL) / 405: índice de resistência à insulina.

    Uma árvore só corta uma coluna por vez, então não acha esse produto sozinha.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["HOMA_IR"] = X["Glucose"] * X["Insulin"] / 405
        return X


def bmi_age_bins():
    return BmiAgeBins()


def ratios():
    return Ratios()


def homa():
    return Homa()
