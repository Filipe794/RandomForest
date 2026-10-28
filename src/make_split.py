"""Gera data/split/train.csv e test.csv.

No projeto Random Forest o split é COPIADO do DecisionTree (mesmo arquivo, mesmo sha256), para
comparar a árvore e a floresta fold a fold. Este script fica só para reprodutibilidade: rodado
com a mesma seed, gera o mesmo split.

Uso: python -m src.make_split [--force]
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import RAW_CSV, TRAIN_CSV, TEST_CSV, COLS, SEED, TEST_SIZE, TARGET


def fix_dpf(df):
    """No CSV, DPF < 1 perdeu o '0.' (0.627 -> 627). O máximo real é 2.42, então >= 3 é sempre um desses."""
    df = df.copy()
    wrong = df["DiabetesPedigreeFunction"] >= 3
    df.loc[wrong, "DiabetesPedigreeFunction"] /= 1000
    print(f"DiabetesPedigreeFunction: {wrong.sum()} valores corrigidos (/1000)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="regera o split (invalida a comparação com execuções antigas)")
    args = ap.parse_args()

    if TRAIN_CSV.exists() and not args.force:
        print(f"Split já existe em {TRAIN_CSV.parent}. Nada feito (use --force para regerar).")
        return

    df = fix_dpf(pd.read_csv(RAW_CSV, header=None, names=COLS))
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=SEED,
                                   stratify=df[TARGET])
    TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)
    print(f"train: {len(train)} linhas | test: {len(test)} linhas")


if __name__ == "__main__":
    main()
