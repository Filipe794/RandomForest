"""OOB x validação cruzada: o erro out-of-bag concorda com o CV 5x5?

Uso: python oob_vs_cv.py

Lê o oob.json e o metrics.json de todas as execuções de results/runs/ e grava
results/oob_vs_cv.csv e results/oob_vs_cv.png (dispersão AUC OOB x AUC do CV).

Ficam de fora da correlação (mas aparecem na tabela):
- execuções sem OOB (balanced_rf, random_under, bootstrap: false);
- execuções calibradas: o OOB é da floresta sem calibração, o CV é da versão calibrada.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.config import OOB_CSV, RUNS_DIR


def main():
    rows = []
    for d in sorted(RUNS_DIR.glob("*/")):
        if d.name.startswith(".tmp_") or not (d / "oob.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        oob = json.loads((d / "oob.json").read_text(encoding="utf-8"))
        rows.append({"execucao": d.name, "name": meta["name"],
                     "cv_roc_auc": metrics["roc_auc"]["mean"],
                     "oob_roc_auc": oob.get("roc_auc", np.nan),
                     "cv_brier": metrics["brier"]["mean"], "oob_brier": oob.get("brier", np.nan),
                     "oob_disponivel": oob.get("disponivel", False),
                     "calibrado": oob.get("calibrado_no_cv", False),
                     "motivo_sem_oob": oob.get("motivo", "")})
    if not rows:
        print("Nenhuma execução com oob.json em results/runs/.")
        return
    df = pd.DataFrame(rows)
    df["dif_auc (oob - cv)"] = df["oob_roc_auc"] - df["cv_roc_auc"]
    OOB_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OOB_CSV, index=False)

    use = df[df["oob_disponivel"] & ~df["calibrado"]]
    print(f"{len(df)} execuções; {len(use)} com OOB comparável ao CV "
          f"({(~df['oob_disponivel']).sum()} sem OOB, "
          f"{(df['oob_disponivel'] & df['calibrado']).sum()} calibradas).")
    if len(use) < 3:
        print("Poucas execuções comparáveis para medir a concordância.")
        return

    d = use["dif_auc (oob - cv)"]
    print(f"\nroc_auc: OOB - CV = {d.mean():+.4f} em média "
          f"(desvio {d.std():.4f}, de {d.min():+.4f} a {d.max():+.4f})")
    print(f"  correlação de Pearson  {pearsonr(use['oob_roc_auc'], use['cv_roc_auc']).statistic:.3f}")
    print(f"  correlação de Spearman {spearmanr(use['oob_roc_auc'], use['cv_roc_auc']).statistic:.3f}"
          "  (ordenam as configs do mesmo jeito?)")
    db = use["oob_brier"] - use["cv_brier"]
    print(f"brier:   OOB - CV = {db.mean():+.4f} em média (desvio {db.std():.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, m in zip(axes, ["roc_auc", "brier"]):
        x, y = use[f"cv_{m}"], use[f"oob_{m}"]
        ax.scatter(x, y, s=18, color="#4C72B0", alpha=0.7)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        pad = 0.05 * (hi - lo or 1)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", color="grey",
                label="OOB = CV")
        ax.set_xlabel(f"{m} na validação cruzada 5x5 (média)")
        ax.set_ylabel(f"{m} out-of-bag (floresta final)")
        ax.set_title(f"{m}: {len(use)} execuções")
        ax.legend()
    fig.tight_layout()
    out = OOB_CSV.with_suffix(".png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n-> {OOB_CSV}\n-> {out}")


if __name__ == "__main__":
    main()
