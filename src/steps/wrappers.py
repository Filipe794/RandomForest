"""Etapas opcionais que envolvem o pipeline inteiro, nesta ordem:

1. calibration: recalibra as probabilidades (CalibratedClassifierCV);
2. threshold: escolhe o limiar de decisão (TunedThresholdClassifierCV), sobre as
   probabilidades já calibradas.

As duas usam um CV interno feito só com o treino recebido.
"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV
from src.config import SEED


def _inner_cv(n_splits):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def calibrated(pipe, method, n_splits=5):
    """ensemble=False: o calibrador é ajustado em previsões fora da amostra (CV interno) e a
    floresta é treinada UMA vez no treino inteiro. Assim continua existindo uma única floresta,
    com as mesmas árvores da versão sem calibração (importâncias, OOB e árvore de exemplo)."""
    return CalibratedClassifierCV(pipe, method=method, cv=_inner_cv(n_splits), ensemble=False)


def sigmoid(pipe, n_splits=5):
    # Método de Platt: ajusta uma curva logística. Mais seguro com poucos dados.
    return calibrated(pipe, "sigmoid", n_splits)


def isotonic(pipe, n_splits=5):
    # Função em degraus, mais flexível; pode sobreajustar com ~430 pacientes por fold.
    return calibrated(pipe, "isotonic", n_splits)


def tuned(pipe, scoring="f1", n_splits=5):
    # Evite scoring="recall": o limiar ótimo seria 0 (tudo positivo)
    return TunedThresholdClassifierCV(pipe, scoring=scoring, cv=_inner_cv(n_splits))
