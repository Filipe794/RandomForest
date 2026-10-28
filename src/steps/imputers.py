"""Etapa 3 (imputer).

Cinco métodos, um de cada família (ver README, seção 4):
  none        sem imputação: a floresta trata os NaN sozinha (scikit-learn >= 1.4)
  constant    ausência como informação: -1, abaixo de qualquer valor real
  median      estatística simples
  knn_scaled  vizinhos (distância com colunas padronizadas)
  missforest  modelo: IterativeImputer com uma Random Forest (Stekhoven e Bühlmann, 2012)
`iterative` (linear) fica disponível só para a comparação opcional "linear x não linear".
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.config import SEED


class KNNScaledImputer(BaseEstimator, TransformerMixin):
    """KNN com colunas padronizadas para escolher os vizinhos.

    O KNNImputer puro mede distância nas unidades originais, então Insulin e Glucose
    dominam. Aqui a distância é calculada em escala padronizada, os valores voltam para
    as unidades originais e só as células ausentes são trocadas (observados ficam idênticos).
    """

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        # Ignora NaN no fit; coluna de desvio zero recebe scale_ = 1 (sem divisão por zero).
        # set_output("default"): internos em numpy, independente do set_config global
        self.scaler_ = StandardScaler().set_output(transform="default").fit(X)
        self.knn_ = (KNNImputer(n_neighbors=self.n_neighbors, keep_empty_features=True)
                     .set_output(transform="default")
                     .fit(self.scaler_.transform(X)))
        return self

    def transform(self, X):
        back = self.scaler_.inverse_transform(self.knn_.transform(self.scaler_.transform(X)))
        mask = X.isna().to_numpy()
        out = X.to_numpy(dtype=float, copy=True)
        out[mask] = back[mask]
        return pd.DataFrame(out, columns=X.columns, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_.copy()


def constant(fill_value=-1):
    # -1 fica abaixo de qualquer valor real: um único corte separa quem não fez o exame
    return SimpleImputer(strategy="constant", fill_value=fill_value)


def median():
    return SimpleImputer(strategy="median")


def knn_scaled(n_neighbors=5):
    return KNNScaledImputer(n_neighbors)


def missforest(n_estimators=50, max_iter=5):
    """MissForest: prevê cada coluna ausente com uma Random Forest treinada nas outras,
    repetindo a rodada de previsões max_iter vezes. É o imputador mais lento (várias florestas
    por fold). Com florestas, o critério de parada do IterativeImputer quase nunca é atingido
    (as previsões oscilam um pouco), então ele faz sempre max_iter rodadas; o MissForest
    original costuma parar entre 4 e 6."""
    return IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=n_estimators, n_jobs=1,
                                        random_state=SEED),
        max_iter=max_iter, random_state=SEED)


def iterative(max_iter=10):
    # Versão linear (BayesianRidge) do mesmo esquema; só para a comparação opcional
    return IterativeImputer(max_iter=max_iter, random_state=SEED)
