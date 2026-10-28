"""Compara execuções fold a fold com o teste t corrigido de Nadeau e Bengio (2003).

Uso:
  python compare.py <pasta_run_A> <pasta_run_B> [--metric roc_auc]
  python compare.py --rodada R1_ [--metric roc_auc] [--ref <pasta>] [--incluir <pasta> ...] [--salvar]
  python compare.py ../DecisionTree/results/runs/<vencedor_DT> <vencedor_RF>   # árvore x floresta

Uma pasta pode ser só o nome (procurada em results/runs/) ou um caminho, inclusive de uma
execução do projeto DecisionTree: o split e os folds são os mesmos.

As notas de k-fold repetido não são independentes (os folds compartilham dados de treino),
então o teste t comum subestima a variância e acha "diferenças" demais. A correção soma
n_val/n_tr à variância da média das diferenças.

Modo rodada: compara a referência (por padrão, a execução de melhor média na métrica: a maior,
ou a menor no brier e no log_loss) contra
cada uma das outras execuções da rodada (um contra todos, não todos os pares). A escolha da
mais simples entre as sem diferença significativa fica com você: o script não escolhe.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

import src  # noqa: F401  aplica set_config(transform_output="pandas") antes de carregar os joblib
from src.builder import unwrap
from src.config import (ALPHA, COMPARACOES_DIR, LOWER_IS_BETTER, N_REPEATS, N_SPLITS,
                        PRIMARY_METRIC, RUNS_DIR, SCORING)
SIG_COL = "significativo (p < α)"


class RunError(ValueError):
    """Execução que não pode entrar na comparação (mensagem pronta para o usuário)."""


def run_path(arg):
    """Aceita o nome de uma pasta em results/runs/ ou um caminho."""
    path = Path(arg)
    return path if path.is_dir() else RUNS_DIR / arg


def read_run(run_dir):
    """Devolve (nome da pasta, sha256 do train.csv, notas por fold); levanta RunError."""
    csv, meta = run_dir / "fold_scores.csv", run_dir / "meta.json"
    if not csv.exists():
        raise RunError(f"Execução '{run_dir.name}' não encontrada (procurei {csv})")
    if not meta.exists():
        raise RunError(f"Execução '{run_dir.name}' sem meta.json: não dá para conferir o split")
    train_sha = json.loads(meta.read_text(encoding="utf-8")).get("train_sha256")
    if not train_sha:
        raise RunError(f"O meta.json de '{run_dir.name}' não tem o campo train_sha256")
    folds = pd.read_csv(csv).sort_values(["repeat", "fold"]).reset_index(drop=True)
    return run_dir.name, train_sha, folds


def load_run(arg):
    """Como read_run, mas para o script com sys.exit (modo de duas execuções)."""
    path = run_path(arg)
    if not (path / "fold_scores.csv").exists():
        sys.exit(f"Execução '{arg}' não encontrada (procurei {path / 'fold_scores.csv'})")
    try:
        return read_run(path)
    except RunError as e:
        sys.exit(str(e))


def corrected_ttest(d, k=N_SPLITS, r=N_REPEATS):
    """t = mean(d) / sqrt((1/(k*r) + n_val/n_tr) * var(d)), com n_val/n_tr = 1/(k-1)."""
    var = np.var(d, ddof=1)
    if var == 0:  # diferença constante em todos os folds
        return (0.0, 1.0) if np.mean(d) == 0 else (np.inf * np.sign(np.mean(d)), 0.0)
    t = np.mean(d) / np.sqrt((1 / (k * r) + 1 / (k - 1)) * var)
    p = 2 * stats.t.sf(abs(t), df=k * r - 1)
    return t, p


def significant(p):
    return "sim" if p < ALPHA else "não"


def compare_two(args):
    """Modo original: duas execuções, todas as métricas (ou só --metric)."""
    name_a, sha_a, a = load_run(args.run_a)
    name_b, sha_b, b = load_run(args.run_b)
    # 6 caracteres: mesmo formato da coluna "split" do summary.csv
    if sha_a != sha_b:
        sys.exit(f"As execuções usam splits diferentes (A: {sha_a[:6]}, B: {sha_b[:6]}). "
                 "Só compare execuções feitas com o mesmo train.csv.")
    n = N_SPLITS * N_REPEATS
    keys_a, keys_b = a[["repeat", "fold"]], b[["repeat", "fold"]]
    if len(a) != n or len(b) != n or not keys_a.equals(keys_b):
        sys.exit(f"As execuções não têm as mesmas {n} linhas (repeat, fold): "
                 f"A tem {len(a)}, B tem {len(b)}. Só compare execuções do mesmo CV.")

    # Execuções do DecisionTree não têm brier/log_loss: compara só as métricas das duas
    wanted = [args.metric] if args.metric else SCORING
    metrics = [m for m in wanted if m in a.columns and m in b.columns]
    if not metrics:
        sys.exit(f"Nenhuma das métricas {wanted} existe nas duas execuções.")
    skipped = [m for m in wanted if m not in metrics]
    rows = []
    for m in metrics:
        d = (b[m] - a[m]).to_numpy()
        t, p = corrected_ttest(d)
        rows.append({"métrica": m, "média A": a[m].mean(), "média B": b[m].mean(),
                     "dif (B-A)": d.mean(), "t": t, "p": p, SIG_COL: significant(p)})

    print(f"A: {name_a}\nB: {name_b}\nsplit: {sha_a[:6]}")
    print(f"Teste t corrigido (Nadeau e Bengio), {N_SPLITS} folds x {N_REPEATS} repetições, "
          f"{n - 1} graus de liberdade\n")
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))
    if skipped:
        print(f"\nMétricas ausentes em uma das execuções (não comparadas): {', '.join(skipped)}")
    low = [m for m in metrics if m in LOWER_IS_BETTER]
    if low:
        print(f"Em {', '.join(low)}, menor é melhor: dif negativa favorece B.")


# ---------------------------------------------------------------- modo rodada

def warn(msg):
    print(f"Aviso: {msg}", file=sys.stderr)


def select_round(prefix):
    """Pastas de results/runs/ cujo campo "name" do meta.json começa com o prefixo."""
    dirs = []
    for run_dir in sorted(RUNS_DIR.glob("*/")):
        meta = run_dir / "meta.json"
        if run_dir.name.startswith(".tmp_") or not meta.exists():
            continue
        if json.loads(meta.read_text(encoding="utf-8")).get("name", "").startswith(prefix):
            dirs.append(run_dir)
    return dirs


def model_size(run_dir):
    """(nº de árvores, profundidade média, folhas médias) do modelo final.

    Lê o forest_stats.json; numa execução do DecisionTree (sem esse arquivo) abre a árvore do
    pipeline.joblib e devolve (1, profundidade, folhas). NaN, com aviso, se nada abrir.
    """
    stats = run_dir / "forest_stats.json"
    if stats.exists():
        s = json.loads(stats.read_text(encoding="utf-8"))
        return s["n_estimators"], s["depth"]["mean"], s["leaves"]["mean"]
    try:
        tree = unwrap(joblib.load(run_dir / "pipeline.joblib")).named_steps["model"]
        return 1, tree.get_depth(), tree.get_n_leaves()
    except Exception as e:  # noqa: BLE001  qualquer falha vira NaN, sem parar a comparação
        warn(f"não consegui ler o modelo de '{run_dir.name}' ({type(e).__name__}: {e}); "
             "árvores, profundidade e folhas ficam NaN")
        return np.nan, np.nan, np.nan


def compare_round(args):
    metric = args.metric or PRIMARY_METRIC
    lower = metric in LOWER_IS_BETTER  # brier, log_loss: a melhor é a de MENOR média
    best = "menor" if lower else "maior"
    n = N_SPLITS * N_REPEATS

    # Execuções da rodada + as extras (--incluir) + a referência explícita (--ref), sem repetir
    dirs = select_round(args.rodada)
    extras = list(args.incluir or []) + ([args.ref] if args.ref else [])
    for arg in extras:
        path = run_path(arg)
        if not path.is_dir():
            sys.exit(f"Execução '{arg}' não encontrada (procurei {path})")
        dirs.append(path)
    unique = {}
    for d in dirs:
        unique.setdefault(d.resolve(), d)
    if not unique:
        sys.exit(f"Nenhuma execução em {RUNS_DIR} com nome começando por '{args.rodada}'.")

    runs = {}  # chave: caminho resolvido; valor: (nome, sha, notas)
    for key, d in unique.items():
        try:
            name, sha, folds = read_run(d)
        except RunError as e:
            warn(f"{e}; pulei essa execução")
            continue
        if metric not in folds.columns:
            warn(f"'{name}' não tem a métrica {metric} no fold_scores.csv; pulei essa execução")
            continue
        runs[key] = (name, sha, folds)

    # Referência: a pedida em --ref ou a de melhor média (só entre as que têm as 25 linhas)
    if args.ref:
        ref_key = run_path(args.ref).resolve()
        if ref_key not in runs:
            sys.exit(f"A referência '{args.ref}' não pôde ser lida (veja o aviso acima).")
        if len(runs[ref_key][2]) != n:
            sys.exit(f"A referência '{runs[ref_key][0]}' não tem as {n} linhas (repeat, fold).")
    else:
        complete = {k: v for k, v in runs.items() if len(v[2]) == n}
        if not complete:
            sys.exit(f"Nenhuma execução com as {n} linhas (repeat, fold) esperadas.")
        pick = min if lower else max
        ref_key = pick(complete, key=lambda k: complete[k][2][metric].mean())
    ref_name, ref_sha, ref = runs[ref_key]
    ref_keys = ref[["repeat", "fold"]]

    def row(name, folds, key):
        trees, depth, leaves = model_size(key)
        return {"execução": name, "média": folds[metric].mean(),
                # np.std (ddof=0): o mesmo desvio do metrics.json e do summary.csv
                "desvio": np.std(folds[metric]),
                "dif (outra - ref)": np.nan, "t": np.nan, "p": np.nan, SIG_COL: "",
                "árvores": trees, "prof. média": depth, "folhas média": leaves}

    ref_row = row(ref_name, ref, ref_key)
    ref_row[SIG_COL] = "referência"
    others = []
    for key, (name, sha, folds) in runs.items():
        if key == ref_key:
            continue
        if sha != ref_sha:
            warn(f"'{name}' usa outro split ({sha[:6]}, a referência usa {ref_sha[:6]}); "
                 "pulei essa execução")
            continue
        if len(folds) != n or not folds[["repeat", "fold"]].equals(ref_keys):
            warn(f"'{name}' não tem as mesmas {n} linhas (repeat, fold) da referência "
                 f"(tem {len(folds)}); pulei essa execução")
            continue
        d = (folds[metric] - ref[metric]).to_numpy()
        t, p = corrected_ttest(d)
        r = row(name, folds, key)
        r.update({"dif (outra - ref)": d.mean(), "t": t, "p": p, SIG_COL: significant(p)})
        others.append(r)

    others.sort(key=lambda r: r["média"], reverse=not lower)
    table = pd.DataFrame([ref_row] + others)

    print(f"Rodada: {args.rodada}  |  métrica: {metric}  |  split: {ref_sha[:6]}")
    print(f"Referência: {ref_name}" + ("" if args.ref else f" ({best} média)"))
    if lower:
        print(f"Em {metric}, menor é melhor.")
    print(f"Teste t corrigido (Nadeau e Bengio), bilateral, {N_SPLITS} folds x {N_REPEATS} "
          f"repetições, graus de liberdade k·r - 1 = {n - 1}, α = {ALPHA}")
    print(f"Um contra todos: {len(others)} comparações com a referência, sem correção para "
          "comparações múltiplas\n")

    show = table.copy()
    # Nomes (e o título da coluna) alinhados à esquerda
    width = max(show["execução"].str.len().max(), len("execução"))
    show["execução"] = show["execução"].str.ljust(width)
    show = show.rename(columns={"execução": "execução".ljust(width)})
    for col in ["média", "desvio", "dif (outra - ref)", "t", "p"]:
        show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    show["árvores"] = show["árvores"].map(lambda v: "NaN" if pd.isna(v) else str(int(v)))
    for col in ["prof. média", "folhas média"]:
        show[col] = show[col].map(lambda v: "NaN" if pd.isna(v) else f"{v:.1f}")
    with pd.option_context("display.max_colwidth", 80, "display.width", 250):
        print(show.to_string(index=False))

    tied = [ref_name] + [r["execução"] for r in others if r[SIG_COL] == "não"]
    print(f"\nSem diferença significativa para a referência (p ≥ {ALPHA}), candidatas à mais "
          f"simples ({len(tied)}, incluindo a referência):")
    for name in tied:
        print(f"  {name}")

    if args.salvar:
        COMPARACOES_DIR.mkdir(parents=True, exist_ok=True)
        out = COMPARACOES_DIR / f"{args.rodada.rstrip('_')}_{metric}.csv"
        table.to_csv(out, index=False)
        print(f"\nTabela salva em {out}")


def main():
    # O console do Windows redirecionado para arquivo pode não codificar "α" e "≥"
    try:
        "α≥".encode(sys.stdout.encoding or "utf-8")
    except UnicodeEncodeError:
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(
        description="Teste t corrigido (Nadeau e Bengio) entre execuções, fold a fold.")
    ap.add_argument("run_a", nargs="?", help="pasta em results/runs/ (ou caminho)")
    ap.add_argument("run_b", nargs="?", help="pasta em results/runs/ (ou caminho)")
    ap.add_argument("--metric", choices=SCORING,
                    help="só esta métrica (padrão: todas com duas execuções; "
                         f"{PRIMARY_METRIC} com --rodada)")
    ap.add_argument("--rodada", metavar="PREFIXO",
                    help="compara todas as execuções cujo nome começa com o prefixo (ex.: R1_)")
    ap.add_argument("--ref", metavar="PASTA",
                    help="referência da rodada (padrão: a de melhor média na métrica)")
    ap.add_argument("--incluir", nargs="+", metavar="PASTA",
                    help="execuções extras na rodada (ex.: o vencedor da rodada anterior)")
    ap.add_argument("--salvar", action="store_true",
                    help="grava a tabela em results/comparacoes/<prefixo>_<métrica>.csv")
    args = ap.parse_args()

    if args.rodada:
        if args.run_a or args.run_b:
            ap.error("use duas pastas OU --rodada, não os dois")
        compare_round(args)
    else:
        if not (args.run_a and args.run_b):
            ap.error("informe duas pastas ou use --rodada PREFIXO")
        if args.ref or args.incluir or args.salvar:
            ap.error("--ref, --incluir e --salvar só valem com --rodada")
        compare_two(args)


if __name__ == "__main__":
    main()
