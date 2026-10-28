# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Projeto de mestrado: Random Forest (scikit-learn) na base Pima Indians Diabetes, com um sistema
declarativo de experimentos. É a continuação do projeto `DecisionTree` (pasta irmã, `../DecisionTree`),
com a mesma arquitetura. A documentação de uso para humanos está em `README.md`, em português;
mantenha o código, os comentários e as mensagens em português.

## Comandos

Todos rodam a partir da raiz, com o `.venv` (Python 3.14):

```bash
pip install -r requirements.txt
python run_experiments.py experiments/r1_num_arvores.yaml                    # uma rodada
python run_experiments.py experiments/r1_num_arvores.yaml --only R1_arvores  # um experimento (ou um sweep inteiro)
python run_experiments.py experiments/r1_num_arvores.yaml --force            # repete, cria pasta nova
python summarize.py [--metric recall]          # regenera results/summary.csv
python compare.py <run_A> <run_B> [--metric roc_auc]   # teste t corrigido (Nadeau-Bengio) fold a fold
python compare.py --rodada R1_ [--metric roc_auc|brier] [--ref <pasta>] [--incluir <pasta> ...] [--salvar]
python final_test.py --run <pasta em results/runs>
# depois do teste final:
python dt_vs_rf.py --rf <vencedor> --dt <vencedor do DT>
python explain.py --run <vencedor>
python stability.py --run <vencedor> [--run-dt <vencedor do DT>] [--seeds R12_]
python oob_vs_cv.py
```

Não há testes, linter nem build. Para validar uma mudança, rode os experimentos afetados com
`--only`. Evite gravar resultados de teste em `results/` do usuário; use uma cópia do projeto.
Para validar um YAML sem rodar nada, use `load_experiments()` + `build_pipeline()` de
`run_experiments.py` num script temporário.

## Arquitetura

O fluxo é YAML → `resolve` → `build_pipeline` → `cross_validate` → pasta imutável em `results/runs/`.

- **Etapas fixas e ordenadas:** `src/registry.py` define `STEP_ORDER` (zeros → missing_flags →
  imputer → features → selection → balancing), seguida do `model` e das `OPTIONAL_STEPS`
  (`calibration`, depois `threshold`), que envolvem o pipeline inteiro (`CalibratedClassifierCV`
  com `ensemble=False`, `TunedThresholdClassifierCV`; ambos em `src/steps/wrappers.py`).
  Cada etapa recebe uma opção pelo nome em `REGISTRY`; `none` pula a etapa.
- **Model:** escrito como `{name: random_forest | balanced_rf, **hiperparâmetros}` (sem `name`,
  vale `random_forest`). As fábricas em `src/steps/models.py` fixam `random_state=SEED`,
  `n_jobs=1` e, na `random_forest`, `oob_score=True` quando há bootstrap.
- **Normalização da config (`src/builder.py:resolve`):** baseline + override (o override
  **substitui a etapa inteira**, inclusive `model`); etapas viram `{name: ..., **params}`;
  parâmetros do model iguais ao padrão da fábrica são removidos (mesma floresta = mesmo hash);
  uma etapa opcional com `none` é removida; as chaves saem em ordem fixa. `expand_sweep`
  chama `resolve` de novo em cada combinação.
- **Método novo:** escreva a função-fábrica em `src/steps/<etapa>.py` e acrescente uma linha no
  `REGISTRY`. O builder, o runner e o YAML não mudam. Transformadores que aprendem algo dos dados
  calculam no `fit` (anti-vazamento). Os customizados devolvem DataFrame e implementam
  `get_feature_names_out` quando necessário (ex.: `KNNScaledImputer`).
- **Não altere o comportamento de uma opção existente:** o hash não inclui o código, então
  execuções antigas ficariam com o mesmo hash e outro resultado. Crie uma opção nova.
- **Compatibilidade com o DecisionTree:** o `stability.py`, o `dt_vs_rf.py` e o `compare.py`
  carregam o `pipeline.joblib` do projeto anterior. O pickle referencia `src.steps.zeros`,
  `src.steps.imputers.KNNScaledImputer` e `src.steps.features`: mantenha esses módulos e nomes
  de classe.
- **Pipeline:** usa `imblearn.pipeline.Pipeline`. O `src/__init__.py` aplica
  `set_config(transform_output="pandas")` globalmente (necessário também ao carregar um
  `pipeline.joblib`) e silencia dois avisos esperados (BalancedRandomForest e o critério de
  parada do `missforest`). O config do sklearn é por thread: em paralelismo próprio, use
  `sklearn.utils.parallel.Parallel/delayed`, que repassam o config às threads.
- **`builder.unwrap()` / `forest_of()`:** chegam ao pipeline e à floresta sob os envoltórios
  (limiar: `estimator_`; calibração: `calibrated_classifiers_[0].estimator`).
- **Runner (`src/runner.py`):**
  - o hash da config completa + seed + CV + métricas + sha256 do `train.csv` define a identidade
    da execução; um hash já existente é pulado;
  - `cross_validate` com `return_estimator` e `return_indices`; depois, para cada fold, MDI da
    floresta e importância por permutação (roc_auc, `PERM_REPEATS`) no fold de validação, sobre
    os exames originais → `fold_importances.csv` (formato longo: repeat, fold, tipo, feature);
  - grava em `.tmp_*` e renomeia no fim; nunca apaga nem edita execuções existentes;
  - floresta final no treino inteiro: `pipeline.joblib`, `forest_stats.json`, `oob.json`,
    `importances_mdi.csv`, `importances_permutation.csv`, `importances.png`, `tree_example.png`.
- **Métricas:** `SCORING` e `SCORERS` em `src/config.py` são a única lista (accuracy, precision,
  recall, f1, balanced_accuracy, roc_auc, brier, log_loss; `PRIMARY_METRIC = roc_auc`). `brier`
  e `log_loss` são funções próprias que devolvem o valor positivo; `LOWER_IS_BETTER` diz ao
  `compare.py` e ao `summarize.py` que, nelas, menor é melhor. Mudar o `SCORING` muda o hash de
  todas as configs.
- **`compare.py`:** igual ao do DecisionTree, com a referência pela menor média nas métricas de
  `LOWER_IS_BETTER`, colunas de tamanho da floresta (lidas do `forest_stats.json`; para uma
  execução do DT, do `pipeline.joblib`) e, no modo de duas execuções, só as métricas presentes
  nas duas. Aceita caminhos de execuções do DecisionTree (mesmo split e mesmos folds).
- **`explain.py`:** PDP próprio (grade sem os zeros = ausentes), SHAP exato intervencional sobre
  os 8 exames originais (2^8 coalizões, fundo de 50 pacientes do treino; sem a biblioteca
  `shap`) e árvore substituta treinada nas previsões da floresta.

## Rodadas de experimentos

Um YAML por rodada em `experiments/`; o vencedor de uma rodada vira o baseline da seguinte.

| Arquivo | Pergunta | Critério |
|---|---|---|
| `r1_num_arvores.yaml` | `n_estimators` | `roc_auc`, o menor sem diferença |
| `r2_max_features.yaml` | `max_features` (sqrt, log2, 0.5, null = bagging) e `max_samples` | `roc_auc`, sqrt no empate |
| `r3_tamanho_arvores.yaml` | `max_depth` x `min_samples_leaf` | `roc_auc` |
| `r4_ausentes.yaml` | imputer (none, constant, median, knn_scaled, missforest) x flags | `roc_auc` |
| `r5_desbalanceamento.yaml` | `class_weight`, `balanced_rf`, `threshold`, `random_under` | F1 ou `balanced_accuracy`, recall ≥ 0.70 |
| `r6_variaveis.yaml` | variáveis derivadas | `roc_auc` |
| `r7_combinacao.yaml` | combinação + ablação | o da rodada que decidiu cada ponto |
| `r8_calibracao.yaml` | `calibration` (sigmoid, isotonic) | `brier` |
| `r12_robustez.yaml` | o vencedor com 5 outras sementes (só depois do teste final) | — |

Em todas: entre as configs sem diferença significativa para a melhor (`compare.py`), fica a mais
simples. R9 (escolha), R10 (`final_test.py`, uma vez) e R11 (`dt_vs_rf.py`, `explain.py`) não têm
YAML.

- Do R2 em diante, linhas com `# ATUALIZAR: vencedor de Rx` são provisórias e devem ser trocadas
  pelo usuário antes de rodar. Não preencha vencedores por conta própria.
- O baseline de cada rodada é a mesma config do vencedor anterior, então o runner o pula pelo
  hash: a referência no `compare.py --rodada` vem por `--incluir <vencedor anterior>`.

## Restrições importantes

- **O conjunto de teste é isolado:** `load_test()` só é chamado pelo `final_test.py` e, depois
  dele, pelo `dt_vs_rf.py` e pelo `explain.py` (que conferem se o teste final daquela execução
  existe). Não rode nenhum deles sem o usuário pedir.
- **Sem SMOTE, por decisão do usuário.** O desbalanceamento é tratado com `class_weight`,
  `balanced_rf`, ajuste de limiar ou `random_under`.
- **Sem etapa de outliers nem de padronização, por decisão do usuário:** árvores só usam a ordem
  dos valores. Uma chave desconhecida no YAML gera erro.
- **CV fixo** em `src/config.py` (`RepeatedStratifiedKFold` 5x5, `SEED=42`), igual ao do
  DecisionTree. Componentes aleatórios devem receber `random_state=SEED`.
- **Split copiado do DecisionTree:** não regere `data/split/` (mudaria o sha256 e quebraria a
  comparação com a árvore).
- **Paralelismo:** a validação cruzada roda com `parallel_config(backend="threading")`. O backend
  padrão do joblib (loky) quebra no Python 3.14 porque o caminho do projeto tem acento.
- **NaN no pipeline (`imputer: none`):** a floresta, o `random_under` e o `balanced_rf` aceitam
  NaN; `selection` (kbest, rfe) não, e o `build_pipeline` rejeita essa combinação.
