"""Árvore de decisão x Random Forest (R11): os dois vencedores lado a lado.

Uso: python dt_vs_rf.py --rf <vencedor RF> --dt <vencedor DT>

--dt aceita o nome da pasta (procurada em ../DecisionTree/results/runs/) ou um caminho.
Os dois precisam ter passado pelo teste final (final_test.py em cada projeto): este script
reavalia os dois modelos salvos no test.csv só para pô-los lado a lado com as MESMAS métricas
(inclusive o Brier, que o DecisionTree não registrava). Nenhuma escolha muda com isso.

Grava em results/dt_vs_rf/<rf>/:
  cv.csv                comparação fold a fold no CV (teste t corrigido de Nadeau e Bengio)
  teste.csv             métricas no teste, os dois modelos
  confusao.csv          matrizes de confusão no teste
  roc_calibracao.png    curvas ROC e de calibração dos dois no teste
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve

import src  # noqa: F401  aplica set_config(transform_output="pandas") antes de carregar os joblib
from compare import SIG_COL, corrected_ttest, read_run, significant
from src.builder import unwrap
from src.config import (DT_PROJECT, DT_VS_RF_DIR, FINAL_DIR, LOWER_IS_BETTER, N_REPEATS,
                        N_SPLITS, RUNS_DIR, SCORERS, SCORING)
from src.data import load_test


def size(pipe, run_dir):
    stats = run_dir / "forest_stats.json"
    if stats.exists():
        s = json.loads(stats.read_text(encoding="utf-8"))
        return (f"{s['n_estimators']} árvores, profundidade média {s['depth']['mean']:.1f}, "
                f"{s['leaves']['mean']:.0f} folhas em média")
    tree = unwrap(pipe).named_steps["model"]
    return f"1 árvore, profundidade {tree.get_depth()}, {tree.get_n_leaves()} folhas"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf", required=True, help="vencedor da floresta (pasta em results/runs/)")
    ap.add_argument("--dt", required=True, help="vencedor do DecisionTree (nome ou caminho)")
    args = ap.parse_args()

    rf_dir = RUNS_DIR / args.rf
    dt_dir = Path(args.dt) if Path(args.dt).is_dir() else DT_PROJECT / "results/runs" / args.dt
    for d in (rf_dir, dt_dir):
        if not (d / "pipeline.joblib").exists():
            sys.exit(f"Execução não encontrada: {d}")
    dt_final = dt_dir.parent.parent / "final" / dt_dir.name
    if not (FINAL_DIR / rf_dir.name / "metrics.json").exists():
        sys.exit("Rode antes o final_test.py do vencedor da floresta.")
    if not (dt_final / "metrics.json").exists():
        sys.exit(f"O vencedor do DecisionTree não passou pelo teste final ({dt_final}).")

    # ---- validação cruzada: mesmos 25 folds
    _, sha_dt, f_dt = read_run(dt_dir)
    _, sha_rf, f_rf = read_run(rf_dir)
    if sha_dt != sha_rf:
        sys.exit(f"Splits diferentes (DT {sha_dt[:6]}, RF {sha_rf[:6]}): copie o split do DecisionTree.")
    if not f_dt[["repeat", "fold"]].equals(f_rf[["repeat", "fold"]]):
        sys.exit("As duas execuções não têm os mesmos folds (repeat, fold).")
    cv_rows = []
    for m in [m for m in SCORING if m in f_dt.columns and m in f_rf.columns]:
        d = (f_rf[m] - f_dt[m]).to_numpy()
        t, p = corrected_ttest(d)
        cv_rows.append({"métrica": m, "árvore": f_dt[m].mean(), "floresta": f_rf[m].mean(),
                        "dif (RF - DT)": d.mean(), "t": t, "p": p, SIG_COL: significant(p)})
    cv = pd.DataFrame(cv_rows)

    # ---- teste: os modelos salvos, com as mesmas métricas
    X_test, y_test = load_test()
    pipes = {"árvore": joblib.load(dt_dir / "pipeline.joblib"),
             "floresta": joblib.load(rf_dir / "pipeline.joblib")}
    test = pd.DataFrame({name: {m: float(s(p, X_test, y_test)) for m, s in SCORERS.items()}
                         for name, p in pipes.items()})
    test["dif (RF - DT)"] = test["floresta"] - test["árvore"]
    test.index.name = "métrica"
    # Conferência: a árvore reavaliada deve bater com o teste final guardado no DecisionTree
    stored = json.loads((dt_final / "metrics.json").read_text(encoding="utf-8"))
    off = [m for m, v in stored.items() if m in test.index and abs(test.loc[m, "árvore"] - v) > 1e-9]
    if off:
        print(f"Aviso: a árvore reavaliada difere do teste guardado em {off} "
              "(versões diferentes de bibliotecas?)", file=sys.stderr)

    cms = {name: confusion_matrix(y_test, p.predict(X_test)) for name, p in pipes.items()}

    out = DT_VS_RF_DIR / rf_dir.name
    out.mkdir(parents=True, exist_ok=True)
    cv.to_csv(out / "cv.csv", index=False)
    test.to_csv(out / "teste.csv")
    pd.concat({n: pd.DataFrame(c, index=["real_0", "real_1"], columns=["pred_0", "pred_1"])
               for n, c in cms.items()}).to_csv(out / "confusao.csv")

    fig, (ax_roc, ax_cal) = plt.subplots(1, 2, figsize=(12, 5.5))
    colors = {"árvore": "#DD8452", "floresta": "#4C72B0"}
    for name, p in pipes.items():
        proba = p.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax_roc.plot(fpr, tpr, color=colors[name],
                    label=f"{name} (AUC {test.loc['roc_auc', name]:.3f})")
        frac, mean_pred = calibration_curve(y_test, proba, n_bins=8, strategy="quantile")
        ax_cal.plot(mean_pred, frac, marker="o", color=colors[name],
                    label=f"{name} (Brier {test.loc['brier', name]:.3f})")
    ax_roc.plot([0, 1], [0, 1], ls="--", color="grey")
    ax_roc.set_xlabel("1 - especificidade")
    ax_roc.set_ylabel("recall (sensibilidade)")
    ax_roc.set_title("Curva ROC no teste")
    ax_roc.legend()
    ax_cal.plot([0, 1], [0, 1], ls="--", color="grey", label="calibração perfeita")
    ax_cal.set_xlabel("probabilidade prevista")
    ax_cal.set_ylabel("fração de diabéticas observada")
    ax_cal.set_title("Curva de calibração no teste")
    ax_cal.legend()
    fig.tight_layout()
    fig.savefig(out / "roc_calibracao.png", dpi=120)
    plt.close(fig)

    print(f"Árvore:   {dt_dir.name}\n          {size(pipes['árvore'], dt_dir)}")
    print(f"Floresta: {rf_dir.name}\n          {size(pipes['floresta'], rf_dir)}")
    print(f"\nValidação cruzada ({N_SPLITS}x{N_REPEATS}, mesmos folds; teste t corrigido):")
    print(cv.to_string(index=False, float_format="%.3f"))
    print("\nTeste (231 pacientes, avaliação única de cada modelo):")
    print(test.to_string(float_format="%.3f"))
    for name, c in cms.items():
        print(f"\nMatriz de confusão, {name}:\n{c}")
    low = [m for m in LOWER_IS_BETTER if m in test.index]
    print(f"\nEm {', '.join(low)}, menor é melhor. -> {out}")


if __name__ == "__main__":
    main()
