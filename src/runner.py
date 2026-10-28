"""Roda a validação cruzada de um experimento e grava numa pasta nova e imutável."""
import hashlib
import json
import platform
import shutil
import time
from datetime import datetime

import imblearn
import joblib
from joblib import parallel_config
from sklearn.utils.parallel import Parallel, delayed  # repassa o set_config às threads
import matplotlib
matplotlib.use("Agg")  # grava PNG sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.tree import plot_tree

from src.builder import build_pipeline, forest_of, unwrap
from src.config import (CV, N_REPEATS, N_SPLITS, PERM_REPEATS, RUNS_DIR, SCORERS, SCORING,
                        SEED, TRAIN_CSV)
from src.data import file_sha256, load_train


def config_hash(cfg, train_sha):
    """Identidade do experimento: config completa + seed + CV + métricas + treino."""
    payload = {"config": cfg, "seed": SEED, "cv": [N_SPLITS, N_REPEATS],
               "scoring": SCORING, "train_sha256": train_sha}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def find_existing(full_hash):
    if not RUNS_DIR.exists():
        return None
    for meta in sorted(RUNS_DIR.glob("*/meta.json")):
        if json.loads(meta.read_text(encoding="utf-8")).get("hash") == full_hash:
            return meta.parent
    return None


# ------------------------------------------------------------------ importâncias por fold

def fold_importances(est, X_val, y_val, i):
    """Importâncias de uma floresta do CV, medidas no fold de validação dela.

    - mdi: importância por impureza, nas colunas que chegam à floresta (depois dos tratamentos);
    - permutation: queda do roc_auc ao embaralhar cada exame ORIGINAL no fold de validação
      (passa por todo o pipeline, então inclui o efeito via flags, HOMA etc.).
    """
    repeat, fold = divmod(i, N_SPLITS)
    forest = forest_of(est)
    rows = [{"repeat": repeat, "fold": fold, "tipo": "mdi", "feature": str(f), "importance": v}
            for f, v in zip(forest.feature_names_in_, forest.feature_importances_)]
    perm = permutation_importance(est, X_val, y_val, scoring="roc_auc",
                                  n_repeats=PERM_REPEATS, random_state=SEED, n_jobs=1)
    rows += [{"repeat": repeat, "fold": fold, "tipo": "permutation", "feature": str(f),
              "importance": v} for f, v in zip(X_val.columns, perm.importances_mean)]
    return rows


# ------------------------------------------------------------------ artefatos da floresta final

def forest_stats(forest):
    depths = np.array([t.get_depth() for t in forest.estimators_])
    leaves = np.array([t.get_n_leaves() for t in forest.estimators_])
    return {"n_estimators": len(forest.estimators_),
            "depth": {"mean": float(depths.mean()), "min": int(depths.min()),
                      "max": int(depths.max())},
            "leaves": {"mean": float(leaves.mean()), "min": int(leaves.min()),
                       "max": int(leaves.max())}}


def oob_summary(final_pipe, y, calibrated):
    """roc_auc e Brier out-of-bag da floresta final, ou o motivo de não haver OOB."""
    inner = unwrap(final_pipe)
    forest = inner.named_steps["model"]
    if "balancing" in inner.named_steps:
        return {"disponivel": False,
                "motivo": "balancing reamostra o treino: as linhas OOB não batem com o y original"}
    if type(forest) is not RandomForestClassifier:  # BalancedRandomForest herda dela
        return {"disponivel": False,
                "motivo": f"{type(forest).__name__}: amostras OOB seguem outra lógica"}
    if not getattr(forest, "oob_score", False) or not hasattr(forest, "oob_decision_function_"):
        return {"disponivel": False, "motivo": "sem bootstrap (bootstrap: false)"}
    proba = forest.oob_decision_function_[:, 1]
    ok = np.isfinite(proba)  # com poucas árvores, alguma paciente pode nunca ficar de fora
    yv = np.asarray(y)[ok]
    return {"disponivel": True, "roc_auc": float(roc_auc_score(yv, proba[ok])),
            "brier": float(brier_score_loss(yv, proba[ok])),
            "n_amostras": int(ok.sum()), "n_sem_oob": int((~ok).sum()),
            # O OOB é sempre da floresta sem calibração; o CV de uma execução calibrada não é
            "calibrado_no_cv": bool(calibrated)}


def save_forest(final_pipe, out_dir, y, fold_imp, calibrated):
    """Grava o pipeline treinado e o que dá para ler de uma floresta."""
    joblib.dump(final_pipe, out_dir / "pipeline.joblib", compress=3)
    forest = forest_of(final_pipe)
    names = [str(c) for c in forest.feature_names_in_]

    (out_dir / "forest_stats.json").write_text(json.dumps(forest_stats(forest), indent=2),
                                               encoding="utf-8")
    (out_dir / "oob.json").write_text(json.dumps(oob_summary(final_pipe, y, calibrated),
                                                 indent=2), encoding="utf-8")

    mdi = (pd.DataFrame({"feature": names, "importance": forest.feature_importances_})
           .sort_values("importance", ascending=False))
    mdi.to_csv(out_dir / "importances_mdi.csv", index=False)
    perm = (fold_imp[fold_imp["tipo"] == "permutation"].groupby("feature")["importance"]
            .agg(["mean", "std"]).sort_values("mean", ascending=False).reset_index())
    perm.to_csv(out_dir / "importances_permutation.csv", index=False)
    fold_imp.to_csv(out_dir / "fold_importances.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.45 * len(names))))
    m = mdi.iloc[::-1]
    axes[0].barh(m["feature"], m["importance"], color="#4C72B0")
    axes[0].set_title("Impureza (MDI), floresta final")
    p = perm.iloc[::-1]
    axes[1].barh(p["feature"], p["mean"], xerr=p["std"], color="#DD8452")
    axes[1].set_title(f"Permutação (queda do roc_auc), média dos {N_SPLITS * N_REPEATS} folds")
    axes[1].axvline(0, color="grey", lw=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "importances.png", dpi=120)
    plt.close(fig)

    # Uma árvore qualquer da floresta, só para ilustrar (ela sozinha NÃO é o modelo)
    fig, ax = plt.subplots(figsize=(22, 8))
    plot_tree(forest.estimators_[0], max_depth=3, feature_names=names, class_names=["0", "1"],
              filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title("Árvore 0 da floresta, desenhada até a profundidade 3 (ilustração)")
    fig.savefig(out_dir / "tree_example.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ execução

def run(name, cfg, force=False, n_jobs=-1):
    train_sha = file_sha256(TRAIN_CSV)
    full_hash = config_hash(cfg, train_sha)
    existing = find_existing(full_hash)
    if existing and not force:
        print(f"[{name}] já executado em {existing.name}, pulando (use --force para repetir)")
        return existing

    X, y = load_train()
    pipe = build_pipeline(cfg)
    t0 = time.time()
    # Threads em vez de processos: no Python 3.14 o joblib (loky) quebra quando a pasta
    # temporária tem acento ("Máquina"). Com ~500 linhas, threads custam o mesmo.
    with parallel_config(backend="threading"):
        res = cross_validate(pipe, X, y, cv=CV, scoring=SCORERS, n_jobs=n_jobs,
                             return_estimator=True, return_indices=True)
        val_idx = res["indices"]["test"]
        imp_rows = Parallel(n_jobs=n_jobs)(
            delayed(fold_importances)(est, X.iloc[val], y.iloc[val], i)
            for i, (est, val) in enumerate(zip(res["estimator"], val_idx)))
    elapsed = time.time() - t0
    fold_imp = pd.DataFrame([r for rows in imp_rows for r in rows])

    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{full_hash[:6]}_{name}"
    final_dir = RUNS_DIR / run_id
    tmp_dir = RUNS_DIR / f".tmp_{run_id}"
    tmp_dir.mkdir(parents=True)
    try:
        (tmp_dir / "config.yaml").write_text(
            yaml.safe_dump({"name": name, **cfg}, sort_keys=False, allow_unicode=True),
            encoding="utf-8")

        metrics = {m: {"mean": float(np.mean(res[f"test_{m}"])),
                       "std": float(np.std(res[f"test_{m}"]))} for m in SCORING}
        (tmp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        n = len(res["fit_time"])
        folds = pd.DataFrame({"repeat": np.arange(n) // N_SPLITS, "fold": np.arange(n) % N_SPLITS,
                              **{m: res[f"test_{m}"] for m in SCORING},
                              "fit_time": res["fit_time"]})
        folds.to_csv(tmp_dir / "fold_scores.csv", index=False)

        # Floresta final: mesmo pipeline treinado no treino inteiro (as 25 do CV são descartadas)
        final_pipe = build_pipeline(cfg).fit(X, y)
        save_forest(final_pipe, tmp_dir, y, fold_imp, calibrated="calibration" in cfg
                    and cfg["calibration"]["name"] != "none")

        meta = {"run_id": run_id, "name": name, "hash": full_hash,
                "date": datetime.now().isoformat(timespec="seconds"),
                "train_sha256": train_sha, "n_train": int(len(X)), "seed": SEED,
                "cv": f"RepeatedStratifiedKFold({N_SPLITS}x{N_REPEATS})",
                "elapsed_s": round(elapsed, 2),
                "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                             "imblearn": imblearn.__version__, "pandas": pd.__version__,
                             "numpy": np.__version__}}
        if hasattr(final_pipe, "best_threshold_"):
            # Limiar escolhido no treino inteiro (é o usado pela floresta salva e pelo final_test)
            meta["threshold"] = float(final_pipe.best_threshold_)
        (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        # Só vira uma execução "válida" depois de tudo gravado
        for tentativa in range(3):
            try:
                tmp_dir.rename(final_dir)
                break
            except PermissionError:  # Windows: antivírus/indexador segurando a pasta
                if tentativa == 2:
                    raise
                time.sleep(1)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    show = ["roc_auc", "recall", "f1", "brier"]
    summary = " | ".join(f"{m}={metrics[m]['mean']:.3f}±{metrics[m]['std']:.3f}" for m in show)
    print(f"[{name}] {summary} ({elapsed:.0f}s)  ->  {run_id}")
    return final_dir
