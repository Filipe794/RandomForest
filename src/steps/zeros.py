"""Etapa 1 (zeros) e etapa 2 (missing_flags)."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import ZERO_COLS


class ZeroToNan(BaseEstimator, TransformerMixin):
    def __init__(self, cols=tuple(ZERO_COLS)):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols = list(self.cols)
        X[cols] = X[cols].replace(0, np.nan)
        return X


class AddMissingFlags(BaseEstimator, TransformerMixin):
    """Acrescenta <col>_missing (0/1). Precisa vir ANTES da imputação."""

    def __init__(self, cols=tuple(ZERO_COLS)):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            X[f"{c}_missing"] = X[c].isna().astype(int)
        return X


def to_nan():
    return ZeroToNan()


def flags():
    return AddMissingFlags()
