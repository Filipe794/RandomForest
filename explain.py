"""Interpretabilidade do vencedor (R11): dependência parcial, SHAP e árvore substituta.

Uso: python explain.py --run <pasta em results/runs> [--n-background 50] [--surrogate-depth 3]

Roda DEPOIS do final_test.py (usa o test.csv para os exemplos e para a fidelidade da árvore
substituta) e não influencia nenhuma escolha do modelo. Grava em results/explain/<run>/:

  pdp.png, pdp.csv              dependência parcial (Friedman, 2001) de cada exame
  shap_summary.png              valores SHAP de todas as pacientes do teste
  shap_values.csv               os valores SHAP, paciente a paciente
  shap_exemplos.csv             explicação de algumas pacientes (acertos e erros)
  surrogate_fidelity.csv        fidelidade da árvore substituta, profundidades 2 a 6
  surrogate_tree.txt / .png     a árvore substituta na profundidade escolhida

Os valores SHAP são calculados aqui mesmo, sem a biblioteca shap: com 8 exames há só 2^8 = 256
coalizões, então dá para calcular os valores de Shapley EXATOS (versão intervencional, como no
KernelSHAP de Lundberg e Lee, 2017). Eles explicam o modelo inteiro (tratamentos + floresta +
calibração) em termos dos 8 exames originais, com 0 significando "exame não medido".
"""
import argparse
import json
import math
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

import src  # noqa: F401  aplica set_config(transform_output="pandas") antes de carregar o joblib
from src.builder import unwrap
from src.config import EXPLAIN_DIR, FEATURES, FINAL_DIR, RUNS_DIR, SEED, ZERO_COLS
from src.data import load_test, load_train


def proba(pipe, X):
    return pipe.predict_proba(X)[:, 1]


# ------------------------------------------------------------------ dependência parcial

def partial_dependence(pipe, X, n_points=25):
    """Para cada exame: fixa o valor em todas as pacientes do treino e tira a média do risco.

    A grade vai do percentil 5 ao 95 dos valores medidos (sem os zeros = ausentes).
    """
    rows = []
    for f in FEATURES:
        vals = X[f][X[f] != 0] if f in ZERO_COLS else X[f]
        grid = np.unique(np.percentile(vals, np.linspace(5, 95, n_points)))
        for g in grid:
            Xc = X.copy()
            Xc[f] = g
            rows.append({"feature": f, "valor": g, "risco_medio": proba(pipe, Xc).mean()})
    return pd.DataFrame(rows)


def plot_pdp(pdp, X, base, out):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    for ax, f in zip(axes.ravel(), FEATURES):
        d = pdp[pdp["feature"] == f]
        ax.plot(d["valor"], d["risco_medio"], color="#4C72B0", lw=2)
        ax.axhline(base, color="grey", ls="--", lw=0.8)
        ax.set_title(f)
    lo, hi = axes[0, 0].get_ylim()  # eixo y compartilhado: traços na mesma altura em todos
    lo -= 0.08 * (hi - lo)
    for ax, f in zip(axes.ravel(), FEATURES):
        vals = X[f][X[f] != 0] if f in ZERO_COLS else X[f]
        ax.plot(vals, np.full(len(vals), lo + 0.03 * (hi - lo)), "|", color="black", alpha=0.25)
        ax.set_ylim(lo, hi)
    for ax in axes[:, 0]:
        ax.set_ylabel("risco médio previsto")
    fig.suptitle("Dependência parcial: risco previsto conforme cada exame varia "
                 "(tracejado = risco médio; traços = pacientes do treino)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ SHAP exato

def shapley_weights(m):
    """Peso de cada tamanho de coalizão |S| na fórmula de Shapley: |S|!(m-|S|-1)!/m!."""
    return np.array([math.factorial(s) * math.factorial(m - s - 1) / math.factorial(m)
                     for s in range(m)])


def exact_shap(pipe, X_explain, background):
    """Valores de Shapley exatos (intervencionais) de cada paciente de X_explain.

    v(S) = risco médio quando os exames em S vêm da paciente e os outros vêm do fundo
    (background). phi_i = soma ponderada de v(S ∪ {i}) - v(S). Por construção,
    soma(phi) = risco da paciente - risco médio do fundo.
    """
    cols = list(X_explain.columns)
    m = len(cols)
    masks = ((np.arange(2 ** m)[:, None] >> np.arange(m)) & 1).astype(bool)  # (256, m)
    sizes = masks.sum(1)
    w = shapley_weights(m)
    B = background.to_numpy(dtype=float)
    nb = len(B)
    phis, v_empty = [], None
    t0 = time.time()
    for k, (_, x) in enumerate(X_explain.iterrows()):
        xv = x.to_numpy(dtype=float)
        # (256 coalizões x nb linhas do fundo, m colunas)
        Z = np.where(masks[:, None, :], xv[None, None, :], B[None, :, :]).reshape(-1, m)
        v = proba(pipe, pd.DataFrame(Z, columns=cols)).reshape(2 ** m, nb).mean(1)
        v_empty = v[0]
        phi = np.zeros(m)
        for i in range(m):
            without = ~masks[:, i]
            S = np.flatnonzero(without)
            with_i = S | (1 << i)
            phi[i] = np.sum(w[sizes[S]] * (v[with_i] - v[S]))
        phis.append(phi)
        if (k + 1) % 25 == 0:
            print(f"  SHAP: {k + 1}/{len(X_explain)} pacientes ({time.time() - t0:.0f}s)")
    return np.array(phis), v_empty


def plot_shap(phi, X, out):
    """Resumo no estilo beeswarm: cada ponto é uma paciente; cor = valor do exame."""
    order = np.argsort(np.abs(phi).mean(0))
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(15, 6), width_ratios=[3, 1])
    rng = np.random.default_rng(SEED)
    for row, j in enumerate(order):
        f = X.columns[j]
        vals = X[f].to_numpy(dtype=float)
        missing = (vals == 0) & (f in ZERO_COLS)
        ranks = pd.Series(vals).where(~missing).rank(pct=True).to_numpy()
        y = row + rng.uniform(-0.3, 0.3, len(vals))
        ax.scatter(phi[~missing, j], y[~missing], c=ranks[~missing], cmap="coolwarm", s=10,
                   vmin=0, vmax=1)
        ax.scatter(phi[missing, j], y[missing], c="black", marker="x", s=12, linewidths=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(range(len(order)), X.columns[order])
    ax.set_xlabel("valor SHAP (efeito no risco previsto)")
    ax.set_title("Cada ponto é uma paciente do teste. Azul = valor baixo, vermelho = alto, "
                 "x preto = exame não medido", fontsize=9)
    axb.barh(range(len(order)), np.abs(phi).mean(0)[order], color="#4C72B0")
    axb.set_yticks(range(len(order)), [])
    axb.set_xlabel("média de |SHAP|")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def describe_patient(x, phi, top=3):
    idx = np.argsort(-np.abs(phi))[:top]
    parts = []
    for j in idx:
        f = x.index[j]
        val = "ausente" if (f in ZERO_COLS and x[f] == 0) else f"{x[f]:g}"
        parts.append(f"{f}={val} ({phi[j]:+.3f})")
    return "; ".join(parts)


def pick_examples(y, p, pred, n=2):
    """Pacientes mais claras de cada tipo: acertos confiantes e erros confiantes."""
    df = pd.DataFrame({"y": np.asarray(y), "p": p, "pred": pred})
    groups = {
        "verdadeiro positivo": df[(df.y == 1) & (df.pred == 1)].nlargest(n, "p"),
        "falso negativo": df[(df.y == 1) & (df.pred == 0)].nsmallest(n, "p"),
        "falso positivo": df[(df.y == 0) & (df.pred == 1)].nlargest(n, "p"),
        "verdadeiro negativo": df[(df.y == 0) & (df.pred == 0)].nsmallest(n, "p"),
    }
    return [(tipo, pos) for tipo, g in groups.items() for pos in g.index]


# ------------------------------------------------------------------ árvore substituta

def surrogate(pipe, X_train, X_test, y_test, depths, chosen, out_dir):
    """Árvore rasa treinada para imitar as previsões da floresta (não o Outcome).

    Usa as mesmas colunas que chegam à floresta (depois dos tratamentos), para que as regras
    sejam comparáveis às da árvore do DecisionTree.
    """
    inner = unwrap(pipe)
    prep = inner[:-1]  # tratamentos (o balanceamento não atua no transform)
    Xt_train, Xt_test = prep.transform(X_train), prep.transform(X_test)
    rf_train, rf_test = pipe.predict(X_train), pipe.predict(X_test)
    rows, trees = [], {}
    for d in depths:
        tree = DecisionTreeClassifier(max_depth=d, min_samples_leaf=5, random_state=SEED)
        tree.fit(Xt_train, rf_train)
        trees[d] = tree
        rows.append({"profundidade": d, "folhas": tree.get_n_leaves(),
                     "fidelidade_treino": accuracy_score(rf_train, tree.predict(Xt_train)),
                     "fidelidade_teste": accuracy_score(rf_test, tree.predict(Xt_test)),
                     "roc_auc_real_teste": roc_auc_score(y_test,
                                                         tree.predict_proba(Xt_test)[:, 1]),
                     "acuracia_real_teste": accuracy_score(y_test, tree.predict(Xt_test))})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "surrogate_fidelity.csv", index=False)

    tree = trees[chosen]
    names = [str(c) for c in Xt_train.columns]
    (out_dir / "surrogate_tree.txt").write_text(
        f"Árvore substituta (profundidade {chosen}): imita as previsões da floresta.\n"
        f"Folhas: [n de pacientes que a floresta chama de 0, n que chama de 1]\n\n"
        + export_text(tree, feature_names=names, show_weights=True), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(max(12, tree.get_n_leaves() * 1.6), 7))
    plot_tree(tree, feature_names=names, class_names=["0", "1"], filled=True, rounded=True,
              fontsize=8, ax=ax)
    ax.set_title(f"Árvore substituta, profundidade {chosen} "
                 f"(fidelidade no teste: {table.set_index('profundidade').loc[chosen, 'fidelidade_teste']:.1%})")
    fig.savefig(out_dir / "surrogate_tree.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return table


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="nome da pasta em results/runs/")
    ap.add_argument("--n-background", type=int, default=50,
                    help="pacientes do treino usadas como fundo no SHAP (padrão: 50)")
    ap.add_argument("--surrogate-depth", type=int, default=3,
                    help="profundidade da árvore substituta desenhada (padrão: 3)")
    args = ap.parse_args()

    run_dir = RUNS_DIR / args.run
    if not (run_dir / "pipeline.joblib").exists():
        sys.exit(f"Execução '{args.run}' não encontrada em {RUNS_DIR}")
    if not (FINAL_DIR / args.run / "metrics.json").exists():
        sys.exit("Rode antes o teste final dessa execução (python final_test.py --run ...): "
                 "o explain.py usa o test.csv e só pode vir depois dele.")
    depths = [2, 3, 4, 5, 6]
    if args.surrogate_depth not in depths:
        sys.exit(f"--surrogate-depth precisa estar em {depths}")

    out_dir = EXPLAIN_DIR / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = joblib.load(run_dir / "pipeline.joblib")
    X_train, _ = load_train()
    X_test, y_test = load_test()

    print("Dependência parcial...")
    base = proba(pipe, X_train).mean()
    pdp = partial_dependence(pipe, X_train)
    pdp.to_csv(out_dir / "pdp.csv", index=False)
    plot_pdp(pdp, X_train, base, out_dir / "pdp.png")

    print(f"SHAP exato ({len(X_test)} pacientes do teste, fundo de {args.n_background})...")
    background = X_train.sample(n=min(args.n_background, len(X_train)), random_state=SEED)
    phi, v_empty = exact_shap(pipe, X_test, background)
    p_test = proba(pipe, X_test)
    gap = np.abs(phi.sum(1) - (p_test - v_empty)).max()  # propriedade de eficiência
    shap_df = pd.DataFrame(phi, columns=X_test.columns, index=X_test.index)
    shap_df.insert(0, "risco_previsto", p_test)
    shap_df.insert(0, "real", y_test.to_numpy())
    shap_df.to_csv(out_dir / "shap_values.csv", index_label="paciente")
    plot_shap(phi, X_test, out_dir / "shap_summary.png")

    pred = pipe.predict(X_test)
    examples = []
    for tipo, pos in pick_examples(y_test, p_test, pred):
        x = X_test.iloc[pos]
        examples.append({"paciente": X_test.index[pos], "tipo": tipo, "real": int(y_test.iloc[pos]),
                         "previsto": int(pred[pos]), "risco_previsto": round(p_test[pos], 3),
                         "risco_medio_fundo": round(v_empty, 3),
                         "principais_fatores": describe_patient(x, phi[pos])})
    pd.DataFrame(examples).to_csv(out_dir / "shap_exemplos.csv", index=False)

    print("Árvore substituta...")
    fid = surrogate(pipe, X_train, X_test, y_test, depths, args.surrogate_depth, out_dir)

    (out_dir / "meta.json").write_text(json.dumps(
        {"run": args.run, "n_background": args.n_background, "risco_medio_treino": base,
         "risco_medio_fundo_shap": v_empty, "erro_max_eficiencia_shap": gap,
         "surrogate_depth": args.surrogate_depth}, indent=2, default=float), encoding="utf-8")

    imp = np.abs(phi).mean(0)
    print("\nImportância SHAP (média de |SHAP| no teste):")
    for j in np.argsort(-imp):
        print(f"  {X_test.columns[j]:>25}: {imp[j]:.3f}")
    print("\nÁrvore substituta:")
    print(fid.to_string(index=False, float_format="%.3f"))
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
