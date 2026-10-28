"""Estabilidade das importâncias (R12): a ordem das variáveis muda entre folds e sementes?

Uso:
  python stability.py --run <vencedor RF> [--run-dt <vencedor DT>] [--seeds R12_]

- Floresta: lê o fold_importances.csv da execução (importâncias de cada uma das 25 florestas do CV).
- Árvore (--run-dt): retreina o pipeline do vencedor do DecisionTree nos MESMOS 25 folds e mede
  as mesmas importâncias. Aceita o nome da pasta (procurada em ../DecisionTree/results/runs/)
  ou um caminho.
- Sementes (--seeds): execuções cujo nome começa com o prefixo (o R12 roda o vencedor com
  outros random_state). Compara as importâncias das florestas finais e o roc_auc entre sementes.

Para cada modelo e tipo de importância (mdi, permutation), mede:
  - variação de cada variável entre os folds (desvio e coeficiente de variação);
  - concordância do ranking entre pares de folds (correlação de Spearman média);
  - em quantos folds a variável mais importante é a mesma.
Grava em results/stability/<run>/.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import joblib
from joblib import parallel_config
from sklearn.utils.parallel import Parallel, delayed  # repassa o set_config às threads
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.inspection import permutation_importance

import src  # noqa: F401  aplica set_config(transform_output="pandas") antes de carregar o joblib
from src.builder import unwrap
from src.config import (CV, DT_PROJECT, N_SPLITS, PERM_REPEATS, RUNS_DIR, SEED, STABILITY_DIR,
                        TRAIN_CSV)
from src.data import file_sha256, load_train


def resolve_run(arg, default_root):
    path = Path(arg)
    return path if path.is_dir() else default_root / arg


# ------------------------------------------------------------------ importâncias da árvore

def dt_fold(pipe_template, X, y, train, val, i):
    repeat, fold = divmod(i, N_SPLITS)
    est = clone(pipe_template).fit(X.iloc[train], y.iloc[train])
    tree = unwrap(est).named_steps["model"]
    rows = [{"repeat": repeat, "fold": fold, "tipo": "mdi", "feature": str(f), "importance": v}
            for f, v in zip(tree.feature_names_in_, tree.feature_importances_)]
    perm = permutation_importance(est, X.iloc[val], y.iloc[val], scoring="roc_auc",
                                  n_repeats=PERM_REPEATS, random_state=SEED, n_jobs=1)
    rows += [{"repeat": repeat, "fold": fold, "tipo": "permutation", "feature": str(f),
              "importance": v} for f, v in zip(X.columns, perm.importances_mean)]
    return rows


def dt_fold_importances(dt_dir, n_jobs):
    meta = json.loads((dt_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("train_sha256") != file_sha256(TRAIN_CSV):
        sys.exit("A execução do DecisionTree usa outro train.csv: os folds não seriam os mesmos.")
    template = joblib.load(dt_dir / "pipeline.joblib")
    X, y = load_train()
    with parallel_config(backend="threading"):
        rows = Parallel(n_jobs=n_jobs)(
            delayed(dt_fold)(template, X, y, tr, va, i)
            for i, (tr, va) in enumerate(CV.split(X, y)))
    return pd.DataFrame([r for rs in rows for r in rs])


# ------------------------------------------------------------------ medidas de estabilidade

def per_feature(imp):
    """Média, desvio e coeficiente de variação de cada variável entre os folds."""
    g = imp.groupby("feature")["importance"]
    out = g.agg(media="mean", desvio="std").reset_index()
    out["coef_variacao"] = out["desvio"] / out["media"].abs().replace(0, np.nan)
    return out.sort_values("media", ascending=False)


def ranking_agreement(wide):
    """wide: linhas = folds (ou sementes), colunas = variáveis."""
    rhos = [spearmanr(wide.iloc[a], wide.iloc[b]).statistic
            for a, b in itertools.combinations(range(len(wide)), 2)]
    top = wide.idxmax(axis=1)
    return {"spearman_medio": float(np.nanmean(rhos)), "spearman_min": float(np.nanmin(rhos)),
            "variavel_mais_importante": top.mode().iloc[0],
            "folds_com_essa_no_topo": f"{(top == top.mode().iloc[0]).sum()}/{len(top)}"}


def summarize(model, imp):
    resumo, por_var = [], []
    for tipo, d in imp.groupby("tipo"):
        wide = d.pivot_table(index=["repeat", "fold"], columns="feature", values="importance")
        resumo.append({"modelo": model, "tipo": tipo, "n_folds": len(wide),
                       **ranking_agreement(wide)})
        pf = per_feature(d)
        pf.insert(0, "tipo", tipo)
        pf.insert(0, "modelo", model)
        por_var.append(pf)
    return resumo, por_var


def plot_boxes(imps, out):
    tipos = ["mdi", "permutation"]
    titles = {"mdi": "Impureza (MDI)", "permutation": "Permutação (queda do roc_auc)"}
    fig, axes = plt.subplots(len(tipos), len(imps), figsize=(7 * len(imps), 9), squeeze=False)
    for col, (model, imp) in enumerate(imps.items()):
        for row, tipo in enumerate(tipos):
            ax = axes[row, col]
            d = imp[imp["tipo"] == tipo]
            order = d.groupby("feature")["importance"].mean().sort_values().index
            ax.boxplot([d.loc[d["feature"] == f, "importance"] for f in order],
                       orientation="horizontal", tick_labels=list(order))
            ax.axvline(0, color="grey", lw=0.8)
            ax.set_title(f"{model}: {titles[tipo]}, 25 folds")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ sementes

def seeds_table(run_dir, prefix):
    dirs = [run_dir]
    for d in sorted(RUNS_DIR.glob("*/")):
        meta = d / "meta.json"
        if (not d.name.startswith(".tmp_") and meta.exists() and d.resolve() != run_dir.resolve()
                and json.loads(meta.read_text(encoding="utf-8")).get("name", "")
                .startswith(prefix)):
            dirs.append(d)
    rows, mdi, perm = [], {}, {}
    for d in dirs:
        cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
        seed = cfg["model"].get("random_state", SEED)
        metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        rows.append({"execucao": d.name, "random_state": seed,
                     "roc_auc_media": metrics["roc_auc"]["mean"],
                     "roc_auc_desvio": metrics["roc_auc"]["std"]})
        mdi[seed] = pd.read_csv(d / "importances_mdi.csv").set_index("feature")["importance"]
        perm[seed] = pd.read_csv(d / "importances_permutation.csv").set_index("feature")["mean"]
    table = pd.DataFrame(rows).sort_values("random_state")
    agree = []
    for tipo, dic in [("mdi", mdi), ("permutation", perm)]:
        wide = pd.DataFrame(dic).T
        agree.append({"modelo": "Random Forest (sementes)", "tipo": tipo,
                      "n_folds": f"{len(wide)} sementes", **ranking_agreement(wide)})
    return table, agree


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="vencedor da floresta (pasta em results/runs/)")
    ap.add_argument("--run-dt", help="vencedor do DecisionTree (nome da pasta ou caminho)")
    ap.add_argument("--seeds", metavar="PREFIXO", help="execuções do R12 (ex.: R12_)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    run_dir = resolve_run(args.run, RUNS_DIR)
    if not (run_dir / "fold_importances.csv").exists():
        sys.exit(f"'{args.run}' não tem fold_importances.csv (procurei em {run_dir})")
    out_dir = STABILITY_DIR / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    imps = {"Random Forest": pd.read_csv(run_dir / "fold_importances.csv")}
    if args.run_dt:
        dt_dir = resolve_run(args.run_dt, DT_PROJECT / "results/runs")
        if not (dt_dir / "pipeline.joblib").exists():
            sys.exit(f"Execução do DecisionTree não encontrada em {dt_dir}")
        print(f"Retreinando a árvore {dt_dir.name} nos 25 folds...")
        imps["Árvore de decisão"] = dt_fold_importances(dt_dir, args.n_jobs)
        imps["Árvore de decisão"].to_csv(out_dir / "fold_importances_dt.csv", index=False)

    resumo, por_var = [], []
    for model, imp in imps.items():
        r, p = summarize(model, imp)
        resumo += r
        por_var += p
    if args.seeds:
        table, agree = seeds_table(run_dir, args.seeds)
        table.to_csv(out_dir / "sementes.csv", index=False)
        resumo += agree
        print("\nroc_auc (CV) por semente da floresta:")
        print(table.to_string(index=False, float_format="%.3f"))
        print(f"variação entre sementes: desvio {table['roc_auc_media'].std():.4f}, "
              f"amplitude {np.ptp(table['roc_auc_media']):.4f}")

    resumo = pd.DataFrame(resumo)
    resumo.to_csv(out_dir / "estabilidade_resumo.csv", index=False)
    pd.concat(por_var).to_csv(out_dir / "estabilidade_por_variavel.csv", index=False)
    plot_boxes(imps, out_dir / "importancias_por_fold.png")

    print("\nConcordância do ranking das variáveis (Spearman entre pares; 1 = mesma ordem):")
    with pd.option_context("display.width", 200):
        print(resumo.to_string(index=False, float_format="%.3f"))
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
