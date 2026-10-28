"""config (dict) -> Pipeline do imblearn."""
import copy

from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from src.registry import OPTIONAL_STEPS, REGISTRY, STEP_ORDER


class ConfigError(ValueError):
    pass


def normalize_step(value):
    """'knn_scaled' -> {'name': 'knn_scaled'};  {'name': 'knn_scaled', 'n_neighbors': 5} fica igual."""
    if value is None:
        return {"name": "none"}
    if isinstance(value, str):
        return {"name": value}
    if isinstance(value, dict) and "name" in value:
        return dict(value)
    raise ConfigError(f"Formato inválido: {value!r}. Use 'nome' ou {{name: nome, ...}}")


def _same(a, b):
    return type(a) is type(b) and a == b if isinstance(a, bool) or isinstance(b, bool) else a == b


def normalize_model(value):
    """O model é escrito como {name: random_forest, n_estimators: 300, ...}.

    Sem `name`, vale random_forest. O nome sempre entra na config normalizada (e no hash).
    Parâmetros iguais ao padrão do modelo (ex.: n_estimators: 100, max_depth: null,
    max_features: sqrt) saem da config: a mesma floresta tem sempre o mesmo hash, escrita
    com ou sem eles, e o runner não a executa duas vezes.
    """
    if value is None:
        value = {}
    if isinstance(value, str):
        value = {"name": value}
    if not isinstance(value, dict):
        raise ConfigError(f"model inválido: {value!r}. Use {{name: random_forest, ...}}")
    name = value.get("name", "random_forest")
    if name not in REGISTRY["model"]:
        raise ConfigError(f"model '{name}' não existe, opções: {', '.join(REGISTRY['model'])}")
    defaults = REGISTRY["model"][name]().get_params()
    params = {k: v for k, v in value.items()
              if k != "name" and not (k in defaults and _same(defaults[k], v))}
    return {"name": name, **params}


def resolve(baseline, override=None):
    """Junta baseline + override. O override SUBSTITUI a etapa inteira."""
    cfg = copy.deepcopy(baseline)
    for key, value in (override or {}).items():
        cfg[key] = copy.deepcopy(value)
    valid = STEP_ORDER + ["model"] + OPTIONAL_STEPS
    unknown = [k for k in cfg if k not in valid]
    if unknown:
        raise ConfigError(f"Etapa '{unknown[0]}' não existe. Etapas: {', '.join(valid)}")
    missing = [s for s in STEP_ORDER + ["model"] if s not in cfg]
    if missing:
        raise ConfigError(f"Config sem as etapas: {', '.join(missing)}")
    # Opcionais só entram na config se escritas (e diferentes de none)
    for step in STEP_ORDER + [s for s in OPTIONAL_STEPS if s in cfg]:
        cfg[step] = normalize_step(cfg[step])
    cfg["model"] = normalize_model(cfg["model"])
    # Opcional com none = não escrita (permite "tirar" o limiar ou a calibração num override)
    for step in OPTIONAL_STEPS:
        if step in cfg and cfg[step]["name"] == "none":
            if len(cfg[step]) > 1:
                raise ConfigError(f"{step} 'none' não aceita parâmetros: {cfg[step]}")
            del cfg[step]
    # Ordem estável das chaves (etapas, model, opcionais): o hash não depende da ordem do YAML
    order = STEP_ORDER + ["model"] + OPTIONAL_STEPS
    return {k: cfg[k] for k in order if k in cfg}


def _make(step, spec, *args):
    """Constrói o passo pelo REGISTRY; devolve None para 'none'."""
    params = dict(spec)
    name = params.pop("name")
    options = REGISTRY[step]
    if name not in options:
        raise ConfigError(f"{step} '{name}' não existe, opções: {', '.join(options)}")
    factory = options[name]
    if factory is None:
        if params:
            raise ConfigError(f"{step} 'none' não aceita parâmetros: {params}")
        return None
    try:
        return factory(*args, **params)
    except TypeError as e:
        raise ConfigError(f"{step} '{name}': parâmetros inválidos {params} ({e})") from e


def build_pipeline(cfg):
    steps = [(s, obj) for s in STEP_ORDER if (obj := _make(s, cfg[s])) is not None]
    if cfg["imputer"]["name"] == "none" and cfg["selection"]["name"] != "none":
        # SelectKBest (f_classif) e RFE não aceitam NaN
        raise ConfigError(f"selection '{cfg['selection']['name']}' exige imputer (não aceita NaN)")
    model = _make("model", cfg["model"])
    # Hiperparâmetro com nome errado só apareceria no fit: confere já
    try:
        model.set_params(**{k: v for k, v in cfg["model"].items() if k != "name"})
        model._validate_params()
    except (ValueError, TypeError) as e:
        raise ConfigError(f"model: hiperparâmetro inválido ({e})") from e
    pipe = Pipeline(steps + [("model", model)])
    for step in OPTIONAL_STEPS:  # calibration, depois threshold
        if step in cfg:
            pipe = _make(step, cfg[step], pipe) or pipe
    return pipe


def unwrap(est):
    """Pipeline treinado, mesmo quando envolto pelo ajuste de limiar e/ou pela calibração."""
    while not isinstance(est, Pipeline):
        if hasattr(est, "calibrated_classifiers_"):     # CalibratedClassifierCV(ensemble=False)
            est = est.calibrated_classifiers_[0].estimator
        elif hasattr(est, "estimator_") and isinstance(est, BaseEstimator):  # TunedThreshold
            est = est.estimator_
        else:
            raise TypeError(f"Não sei desembrulhar {type(est).__name__}")
    return est


def forest_of(fitted):
    """A floresta (último passo) de um pipeline treinado, embrulhado ou não."""
    return unwrap(fitted).named_steps["model"]
