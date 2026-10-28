import warnings

from sklearn import set_config

# Transformadores devolvem DataFrame => colunas mantêm o nome entre as etapas.
# Fica aqui para valer em qualquer import de src (inclusive ao carregar um pipeline.joblib).
set_config(transform_output="pandas")

# Aviso interno do imblearn (a BalancedRandomForest repassa numpy às próprias árvores): não
# indica problema nos nossos dados e polui a saída de cada fold.
warnings.filterwarnings("ignore", message="X does not have valid feature names, but "
                        "BalancedRandomForestClassifier")
# O missforest sempre faz max_iter rodadas (ver src/steps/imputers.py): o aviso é esperado.
warnings.filterwarnings("ignore", message=r"\[IterativeImputer\] Early stopping criterion not reached")
