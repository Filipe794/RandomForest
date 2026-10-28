import hashlib
import pandas as pd
from src.config import TRAIN_CSV, TEST_CSV, TARGET


def _load(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe. Copie data/split/ do projeto DecisionTree "
                                "(mesmo split = comparação justa entre os dois modelos)")
    df = pd.read_csv(path)
    return df.drop(TARGET, axis=1), df[TARGET]


def load_train():
    return _load(TRAIN_CSV)


def load_test():  # SOMENTE final_test.py e as análises posteriores a ele (explain.py, dt_vs_rf.py)
    return _load(TEST_CSV)


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
