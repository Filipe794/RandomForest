"""Avaliação final no conjunto de teste (rodar só no fim, uma vez, na execução escolhida).

Uso: python final_test.py --run 20261105_143012_a3f9c1_R7_combinacao
"""
import argparse
import json
import sys
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

import src  # noqa: F401  aplica set_config(transform_output="pandas") antes de carregar o joblib
from src.builder import build_pipeline
from src.config import FINAL_DIR, RUNS_DIR, SCORERS
from src.data import load_test, load_train


def plot_calibration(y, proba, out):
    """Curva de calibração (reliability diagram): em cada faixa de probabilidade prevista,
    a fração de diabéticas de fato. Um modelo calibrado fica sobre a diagonal."""
    frac, mean_pred = calibration_curve(y, proba, n_bins=8, strategy="quantile")
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6, 7), height_ratios=[3, 1], sharex=True)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="calibração perfeita")
    ax.plot(mean_pred, frac, marker="o", color="#4C72B0", label="modelo (teste)")
    ax.set_ylabel("fração de positivas observada")
    ax.set_title("Curva de calibração no teste (8 faixas com o mesmo nº de pacientes)")
    ax.legend(loc="upper left")
    axh.hist(proba, bins=20, range=(0, 1), color="#4C72B0", alpha=0.7)
    axh.set_xlabel("probabilidade prevista de diabetes")
    axh.set_ylabel("pacientes")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="nome da pasta em results/runs/")
    args = ap.parse_args()

    run_dir = RUNS_DIR / args.run
    if not (run_dir / "config.yaml").exists():
        sys.exit(f"Execução '{args.run}' não encontrada em {RUNS_DIR}")
    out_dir = FINAL_DIR / args.run
    if out_dir.exists():
        print(f"Já avaliado no teste. Resultado guardado em {out_dir}:")
        print((out_dir / "metrics.json").read_text(encoding="utf-8"))
        return

    X_train, y_train = load_train()
    X_test, y_test = load_test()
    if (run_dir / "pipeline.joblib").exists():
        # Avalia exatamente a floresta salva na execução
        pipe = joblib.load(run_dir / "pipeline.joblib")
    else:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        cfg.pop("name", None)
        pipe = build_pipeline(cfg).fit(X_train, y_train)
    # Mesmas métricas e mesmos scorers da validação cruzada (src/config.py:SCORERS)
    metrics = {m: float(scorer(pipe, X_test, y_test)) for m, scorer in SCORERS.items()}
    cm = confusion_matrix(y_test, pipe.predict(X_test))

    out_dir.mkdir(parents=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(cm, index=["real_0", "real_1"], columns=["pred_0", "pred_1"]).to_csv(
        out_dir / "confusion_matrix.csv")
    plot_calibration(y_test, pipe.predict_proba(X_test)[:, 1], out_dir / "calibration_curve.png")
    meta = {"run": args.run, "date": datetime.now().isoformat(timespec="seconds"),
            "n_train": len(X_train), "n_test": len(X_test)}
    if hasattr(pipe, "best_threshold_"):
        meta["threshold"] = float(pipe.best_threshold_)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for m, v in metrics.items():
        print(f"{m:>17}: {v:.3f}")
    print(f"Matriz de confusão:\n{cm}\n-> {out_dir}")


if __name__ == "__main__":
    main()
