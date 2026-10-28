"""Etapa 6 (balancing). Sempre a última antes do modelo; só atua no fit.

Sem SMOTE nem outras amostras sintéticas (decisão do projeto). Só a subamostragem
aleatória da classe majoritária.
"""
from imblearn.under_sampling import RandomUnderSampler
from src.config import SEED


def random_under():
    return RandomUnderSampler(random_state=SEED)
