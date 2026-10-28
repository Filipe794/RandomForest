"""Catálogo: nome usado no YAML -> função que constrói o passo.

Método novo = função no arquivo da etapa + uma linha aqui.
"""
from src.steps import balancing, features, imputers, models, selection, wrappers, zeros

# Ordem fixa do pipeline. NÃO alterar por experimento.
# Sem etapa de outliers nem de padronização: as árvores só usam a ordem dos valores.
STEP_ORDER = ["zeros", "missing_flags", "imputer",
              "features", "selection", "balancing"]  # + "model" no fim
# Etapas que podem ser omitidas no YAML (equivalem a none). Envolvem o pipeline inteiro,
# nesta ordem: primeiro a calibração, depois o limiar (escolhido sobre as probabilidades calibradas).
OPTIONAL_STEPS = ["calibration", "threshold"]

REGISTRY = {
    "zeros": {
        "none": None,
        "to_nan": zeros.to_nan,
    },
    "missing_flags": {
        "none": None,
        "flags": zeros.flags,
    },
    "imputer": {
        "none": None,
        "constant": imputers.constant,
        "median": imputers.median,
        "knn_scaled": imputers.knn_scaled,
        "missforest": imputers.missforest,
        "iterative": imputers.iterative,
    },
    "features": {
        "none": None,
        "bmi_age_bins": features.bmi_age_bins,
        "ratios": features.ratios,
        "homa": features.homa,
    },
    "selection": {
        "none": None,
        "kbest": selection.kbest,
        "rfe": selection.rfe,
    },
    "balancing": {
        "none": None,
        "random_under": balancing.random_under,
    },
    "model": {
        "random_forest": models.random_forest,
        "balanced_rf": models.balanced_rf,
    },
    "calibration": {
        "none": None,
        "sigmoid": wrappers.sigmoid,
        "isotonic": wrappers.isotonic,
    },
    "threshold": {
        "none": None,
        "tuned": wrappers.tuned,
    },
}
