"""Junta todas as execuções de results/runs/ numa tabela (sempre regenerada).

Uso: python summarize.py [--metric roc_auc]
"""
import argparse
import json

import pandas as pd
import yaml

from src.config import LOWER_IS_BETTER, PRIMARY_METRIC, RUNS_DIR, SCORING, SUMMARY_CSV
from src.registry import OPTIONAL_STEPS, STEP_ORDER


def describe(cfg):
    """Resumo curto: só as etapas diferentes de none.

    Ex.: 'to_nan > median | random_forest(n_estimators=300,max_features=sqrt) > sigmoid'.
    """
    def fmt(spec):
        params = dict(spec)
        name = params.pop("name")
        args = ",".join(f"{k}={v}" for k, v in params.items())
        return f"{name}({args})" if args else name

    parts = [fmt(cfg[s]) for s in STEP_ORDER if cfg[s]["name"] != "none"]
    wrappers = [fmt(cfg[s]) for s in OPTIONAL_STEPS if s in cfg and cfg[s]["name"] != "none"]
    return " > ".join(parts) + " | " + " > ".join([fmt(cfg["model"])] + wrappers)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default=PRIMARY_METRIC, choices=SCORING)
    args = ap.parse_args()

    rows = []
    for run_dir in sorted(RUNS_DIR.glob("*/")):
        if run_dir.name.startswith(".tmp_") or not (run_dir / "meta.json").exists():
            continue
        meta = read_json(run_dir / "meta.json")
        metrics = read_json(run_dir / "metrics.json")
        oob = read_json(run_dir / "oob.json")
        stats = read_json(run_dir / "forest_stats.json")
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        row = {"run_id": meta["run_id"], "name": meta["name"], "date": meta["date"]}
        for m in SCORING:
            # Execuções feitas antes de uma métrica entrar no SCORING ficam com NaN nela
            row[f"{m}_mean"] = metrics.get(m, {}).get("mean", float("nan"))
            row[f"{m}_std"] = metrics.get(m, {}).get("std", float("nan"))
        row["oob_roc_auc"] = oob.get("roc_auc", float("nan"))
        row["arvores"] = stats.get("n_estimators", float("nan"))
        row["prof_media"] = stats.get("depth", {}).get("mean", float("nan"))
        row["folhas_media"] = stats.get("leaves", {}).get("mean", float("nan"))
        row["split"] = meta["train_sha256"][:6]  # distingue execuções feitas com splits diferentes
        row["threshold"] = meta.get("threshold", 0.5)
        row["segundos"] = meta.get("elapsed_s", float("nan"))
        row["pipeline"] = describe(cfg)
        rows.append(row)

    if not rows:
        print("Nenhuma execução em results/runs/.")
        return

    df = pd.DataFrame(rows).sort_values(f"{args.metric}_mean",
                                        ascending=args.metric in LOWER_IS_BETTER)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUMMARY_CSV, index=False)

    show = (["name"] + [f"{m}_mean" for m in ["roc_auc", "recall", "f1", "balanced_accuracy",
                                              "brier"]]
            + [f"{args.metric}_std", "oob_roc_auc", "arvores", "prof_media", "threshold",
               "pipeline"])
    show = list(dict.fromkeys([show[0], f"{args.metric}_mean"] + show[1:]))
    with pd.option_context("display.max_colwidth", 80, "display.width", 250):
        print(df[show].to_string(index=False, float_format="%.3f"))
    print(f"\n{len(df)} execuções -> {SUMMARY_CSV} (a tabela completa tem todas as métricas)")


if __name__ == "__main__":
    main()
