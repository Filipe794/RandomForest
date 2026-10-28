"""Etapa 7 (model): a floresta.

Os parâmetros escritos no YAML vão direto para o construtor. `random_state` é fixado em
SEED, mas pode ser trocado no YAML (é o que o R12 faz para medir a robustez).
`n_jobs=1`: o paralelismo fica na validação cruzada (paralelizar os dois níveis sobrecarrega a CPU).
"""
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from src.config import SEED


def random_forest(**params):
    """RandomForestClassifier (Breiman, 2001).

    O OOB fica ligado sempre que há bootstrap: custa pouco e alimenta o oob_vs_cv.py.
    Não muda as previsões da floresta.
    """
    params.setdefault("random_state", SEED)
    params.setdefault("n_jobs", 1)
    params.setdefault("oob_score", params.get("bootstrap", True))
    return RandomForestClassifier(**params)


def balanced_rf(**params):
    """BalancedRandomForestClassifier (imblearn): cada árvore recebe um bootstrap com as
    classes equilibradas por subamostragem da majoritária (Chen, Liaw e Breiman, 2004).

    Padrões fixados explicitamente (os do imblearn mudaram entre versões): todas as classes
    reamostradas com reposição e sem um segundo bootstrap por cima.
    """
    params.setdefault("random_state", SEED)
    params.setdefault("n_jobs", 1)
    params.setdefault("sampling_strategy", "all")
    params.setdefault("replacement", True)
    params.setdefault("bootstrap", False)
    return BalancedRandomForestClassifier(**params)
