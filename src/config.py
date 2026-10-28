from pathlib import Path

from sklearn.metrics import brier_score_loss, get_scorer, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold

# Raiz do projeto calculada a partir deste arquivo: funciona local e no Colab
BASE = Path(__file__).resolve().parents[1]
RAW_CSV = BASE / "data/raw/pima-indians-diabetes.csv"
TRAIN_CSV = BASE / "data/split/train.csv"
TEST_CSV = BASE / "data/split/test.csv"
RUNS_DIR = BASE / "results/runs"
FINAL_DIR = BASE / "results/final"
SUMMARY_CSV = BASE / "results/summary.csv"
COMPARACOES_DIR = BASE / "results/comparacoes"
EXPLAIN_DIR = BASE / "results/explain"
STABILITY_DIR = BASE / "results/stability"
DT_VS_RF_DIR = BASE / "results/dt_vs_rf"
OOB_CSV = BASE / "results/oob_vs_cv.csv"
# Projeto anterior (árvore de decisão), ao lado deste. Usado pelo stability.py e pelo dt_vs_rf.py
DT_PROJECT = BASE.parent / "DecisionTree"

SEED = 42
TEST_SIZE = 0.3

COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
TARGET = "Outcome"
FEATURES = [c for c in COLS if c != TARGET]
# Colunas em que 0 é fisiologicamente impossível => valor ausente
ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Mesmos folds do DecisionTree => a melhor árvore e a melhor floresta podem ser comparadas fold a fold
N_SPLITS, N_REPEATS = 5, 5
CV = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)


def _proba1(est, X):
    return est.predict_proba(X)[:, 1]


def brier(est, X, y):
    """Brier score: erro quadrático médio da probabilidade prevista (menor é melhor)."""
    return brier_score_loss(y, _proba1(est, X))


def logloss(est, X, y):
    """Log loss (entropia cruzada) da probabilidade prevista (menor é melhor)."""
    return log_loss(y, _proba1(est, X), labels=[0, 1])


# balanced_accuracy = média do recall das duas classes (critério do R5, junto com o F1).
# brier e log_loss medem a qualidade das probabilidades (critério do R8).
SCORING = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "roc_auc",
           "brier", "log_loss"]
# Nome da métrica -> função (est, X, y). A única lista usada pelo runner, final_test e afins.
SCORERS = {m: get_scorer(m) for m in SCORING if m not in ("brier", "log_loss")}
SCORERS.update({"brier": brier, "log_loss": logloss})
# Métricas em que menor é melhor (o compare.py e o summarize.py invertem a ordenação)
LOWER_IS_BETTER = {"brier", "log_loss"}
PRIMARY_METRIC = "roc_auc"

# Importância por permutação: repetições por coluna, em cada fold de validação
PERM_REPEATS = 5

# Nível de significância do teste t corrigido (bilateral) usado pelo compare.py.
# Não entra no hash das execuções: mudar o ALPHA só muda a leitura das comparações.
ALPHA = 0.05
