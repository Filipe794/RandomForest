# Random Forest para triagem de diabetes (Pima Indians Diabetes)

Trabalho da disciplina Aprendizado de Máquina, do PPGCC - UFPI, 2026.2.

**Autores:** Filipe Mendes, Bétanny Alexandra, Juciane Maciel · **Orientação/professor:** Prof. Dr. Vinicius Ponte Machado.

## Objetivo

Construir e avaliar uma Random Forest que identifique pacientes com diabetes a partir de
exames clínicos simples e compará-la com a árvore de decisão do trabalho anterior
(projeto `DecisionTree`). O trabalho investiga, de forma controlada, o efeito do número de
árvores, da aleatoriedade entre elas, do tamanho de cada árvore, do tratamento de valores
ausentes, da correção do desbalanceamento (sem amostras sintéticas), de variáveis derivadas
e da calibração das probabilidades, sempre com validação cruzada e um conjunto de teste
isolado até o fim.

A pergunta de fundo é o **trade-off entre desempenho e interpretabilidade**: a floresta
costuma acertar mais que uma árvore única, mas perde as regras legíveis. Por isso, além do
desempenho, o trabalho avalia:

- **Calibração:** as probabilidades da floresta podem ser lidas como risco real?
- **Interpretabilidade:** dá para recuperar explicações (dependência parcial, SHAP, árvore
  substituta) sem perder o desempenho?
- **Estabilidade:** a importância das variáveis se mantém entre folds e sementes?
- **OOB:** o erro *out-of-bag*, calculado "de graça" pela floresta, concorda com a
  validação cruzada?

## Base de dados

**Pima Indians Diabetes Database:** 768 mulheres de ascendência Pima, com 21 anos ou mais,
descritas por 8 variáveis clínicas (gestações, glicose, pressão arterial, espessura da prega
cutânea, insulina, IMC, histórico familiar e idade). O alvo (`Outcome`) indica diabetes;
cerca de 35% das pacientes são positivas.

- Artigo original: Smith, J. W.; Everhart, J. E.; Dickson, W. C.; Knowler, W. C.; Johannes,
  R. S. *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus*.
  Proceedings of the Symposium on Computer Applications and Medical Care, 1988, p. 261–265.
- Disponível em: [Kaggle — Pima Indians Diabetes Database](https://www.kaggle.com/datasets/kumargh/pimaindiansdiabetescsv/)
  (originalmente no UCI Machine Learning Repository).
- **O split treino/teste é o mesmo do DecisionTree** (os arquivos de `data/split/` foram
  copiados, com o mesmo sha256). Com o mesmo `train.csv` e a mesma validação cruzada, a
  melhor árvore e a melhor floresta podem ser comparadas fold a fold (rodada R11).
- A correção da `DiabetesPedigreeFunction` (valores < 1 que perderam o "0.") já está aplicada
  no split, como no projeto anterior.

## Licença

MIT. A base de dados segue os termos da fonte original.

---

# Random Forest em poucas palavras

Uma árvore de decisão é **um** modelo que faz perguntas em sequência ("glicose > 127?",
"IMC > 30?"). Ela é muito sensível aos dados de treino: trocar algumas pacientes pode mudar
a árvore inteira.

A Random Forest treina **centenas de árvores** e faz a média dos votos. Para que as
árvores não fiquem iguais, cada uma recebe duas fontes de aleatoriedade:

1. **Bootstrap:** cada árvore treina com um sorteio, com reposição, das pacientes do
   treino. Cerca de 1/3 fica de fora de cada árvore e serve para medir o erro *out-of-bag*
   (OOB).
2. **Sorteio de variáveis:** em cada corte, a árvore só pode escolher entre algumas
   variáveis sorteadas (`max_features`). Sem isso, quase todas começariam pela glicose.

Cada árvore erra bastante, mas de um jeito diferente das outras; na média, os erros se
cancelam. Por isso as árvores da floresta normalmente crescem completas, sem poda, que é o
oposto do que foi necessário no trabalho anterior.

| | Árvore de decisão | Random Forest |
|---|---|---|
| Modelo | 1 árvore | centenas de árvores votando |
| Dados de cada árvore | o treino todo | um sorteio (bootstrap) |
| Variáveis em cada corte | todas | um subconjunto sorteado |
| Tamanho ideal da árvore | rasa (poda) | geralmente completa |
| Interpretação | regras legíveis | importância das variáveis, PDP, SHAP |

---

# Como usar o projeto

O projeto é um sistema de experimentos em que cada método de tratamento é uma peça
intercambiável e cada experimento é só uma configuração no YAML. Nenhum resultado é
sobrescrito: cada execução ganha uma pasta própria. É o mesmo sistema do DecisionTree.

## 1. Instalação

Requer **Python 3.14**. O `requirements.txt` é o mesmo do DecisionTree e fixa as versões: o
`pipeline.joblib` de cada execução só abre com as mesmas versões do scikit-learn e do
imbalanced-learn. A floresta só aceita NaN (opção `imputer: none`) a partir do
scikit-learn 1.4; o projeto usa a 1.9.1.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Não é preciso instalar a biblioteca `shap`: os valores SHAP são calculados pelo próprio
`explain.py` (seção 11).

**Paralelismo:** a validação cruzada roda em paralelo com threads
(`parallel_config(backend="threading")`), como no projeto anterior: o backend padrão do
joblib quebra no Python 3.14 quando o caminho tem acento ("Máquina"). Dentro do CV, cada
floresta usa `n_jobs=1`; paralelizar os dois níveis ao mesmo tempo sobrecarrega a CPU.

No Colab, monte o Drive, entre na pasta do projeto e rode os mesmos comandos com `!`:

```python
%cd "/content/drive/MyDrive/Mestrado/Aprendizado de Máquina/RandomForest"
!pip install -r requirements.txt
!python run_experiments.py experiments/r1_num_arvores.yaml
```

Todos os comandos devem ser executados **a partir da raiz do projeto**.

## 2. Rotina de uso

```bash
# 1. atualizar o baseline da rodada (linhas "# ATUALIZAR") e executar
python run_experiments.py experiments/r1_num_arvores.yaml

# 2. comparar as execuções
python summarize.py                                  # tabela geral, ordenada por roc_auc
python compare.py --rodada R1_ --salvar              # fecha a rodada (teste t corrigido)

# 3. no fim, avaliar a melhor no teste (uma vez)
python final_test.py --run <pasta da execução escolhida em results/runs/>

# 4. análises do vencedor (depois do teste final)
python dt_vs_rf.py --rf <vencedor> --dt <vencedor do DT>     # árvore x floresta
python explain.py --run <vencedor>                           # PDP, SHAP e árvore substituta
python run_experiments.py experiments/r12_robustez.yaml      # outras sementes
python stability.py --run <vencedor> --run-dt <vencedor do DT> --seeds R12_
python oob_vs_cv.py                                          # OOB x validação cruzada
```

Opções do `run_experiments.py`:

| Opção | Efeito |
|---|---|
| `--only R2_max_features` | roda só esses experimentos (o nome de um sweep roda todos os seus valores) |
| `--force` | roda de novo mesmo se já executado (cria pasta nova, não substitui) |
| `--n-jobs 1` | número de threads da validação cruzada (padrão: todos os núcleos) |

**Tempo:** com 300 árvores, uma execução simples leva por volta de meio minuto num computador
de 8 núcleos. As mais lentas são as de 1000 árvores (R1), o `missforest` (R4) e as que usam
`threshold` ou `calibration` (CV interno: ~5 vezes mais florestas).

## 3. Estrutura de pastas

```
RandomForest/
├── data/
│   ├── raw/pima-indians-diabetes.csv     nunca alterar
│   └── split/train.csv, test.csv         copiados do DecisionTree (mesmo sha256)
├── src/
│   ├── config.py        caminhos, SEED, CV (5 folds x 5 repetições), métricas, ALPHA
│   ├── data.py          load_train(), load_test()
│   ├── make_split.py    só para reprodutibilidade (o split é o do DecisionTree)
│   ├── steps/           um arquivo por etapa
│   │   ├── zeros.py         zeros -> NaN e colunas de flag
│   │   ├── imputers.py      none, constant, median, knn_scaled, missforest (+ iterative)
│   │   ├── features.py      homa, ratios, bmi_age_bins
│   │   ├── selection.py     kbest, rfe (não usadas nas rodadas)
│   │   ├── balancing.py     random_under
│   │   ├── models.py        random_forest, balanced_rf
│   │   └── wrappers.py      calibration (sigmoid, isotonic) e threshold (tuned)
│   ├── registry.py      catálogo: nome no YAML -> função que constrói o passo
│   ├── builder.py       config -> Pipeline
│   └── runner.py        validação cruzada + importâncias por fold + gravação dos resultados
├── experiments/
│   ├── r1_num_arvores.yaml ... r8_calibracao.yaml   uma rodada por arquivo (seção 7)
│   └── r12_robustez.yaml                            outras sementes (depois do teste)
├── docs/anotacoes_artigo.md   decisões, achados e limitações (para o artigo)
├── results/
│   ├── runs/          uma pasta por execução (só se acrescenta)
│   ├── final/         avaliações no teste
│   ├── comparacoes/   tabelas do compare.py --salvar
│   ├── dt_vs_rf/      árvore x floresta (R11)
│   ├── explain/       PDP, SHAP e árvore substituta do vencedor (R11)
│   ├── stability/     estabilidade das importâncias (R12)
│   ├── oob_vs_cv.csv  OOB x validação cruzada de todas as execuções
│   └── summary.csv    tabela gerada pelo summarize.py
├── run_experiments.py
├── summarize.py
├── compare.py
├── final_test.py
├── dt_vs_rf.py
├── explain.py
├── stability.py
├── oob_vs_cv.py
├── requirements.txt
└── CLAUDE.md          instruções para o Claude Code
```

O `dt_vs_rf.py` e o `stability.py` procuram o projeto anterior em `../DecisionTree` (as duas
pastas lado a lado, como estão hoje). Também aceitam um caminho completo.

No dia a dia só se mexe em dois lugares: os YAMLs das rodadas em `experiments/` (para usar
métodos) e `src/steps/` + `src/registry.py` (para criar métodos novos).

## 4. O pipeline

Toda execução passa pelas mesmas etapas, sempre nesta ordem. Cada etapa recebe uma opção,
e `none` pula a etapa.

| # | Etapa | Opções | Observação |
|---|---|---|---|
| 1 | `zeros` | `none`, `to_nan` | 0 em Glucose, BloodPressure, SkinThickness, Insulin e BMI vira NaN |
| 2 | `missing_flags` | `none`, `flags` | cria colunas `<col>_missing` (0/1) antes da imputação |
| 3 | `imputer` | `none`, `constant`, `median`, `knn_scaled`, `missforest`, `iterative` | preenche os NaN; com `none`, a própria floresta trata os NaN |
| 4 | `features` | `none`, `bmi_age_bins`, `ratios`, `homa` | cria colunas novas |
| 5 | `selection` | `none`, `kbest`, `rfe` | não usada nas rodadas (a floresta já sorteia variáveis); exige imputer |
| 6 | `balancing` | `none`, `random_under` | atua só no treino; sem SMOTE |
| 7 | `model` | `random_forest`, `balanced_rf` + hiperparâmetros | `random_state` fixado em 42 (o R12 troca) |
| — | `calibration` | `none`, `sigmoid`, `isotonic` | **opcional**: recalibra as probabilidades (R8) |
| — | `threshold` | `none`, `tuned` | **opcional**: escolhe o limiar de decisão (padrão 0.5) |

As duas etapas opcionais envolvem o pipeline inteiro, nesta ordem: primeiro a calibração,
depois o limiar (escolhido sobre as probabilidades já calibradas). As duas usam um CV
interno feito só no treino.

- `calibration`: usa o `CalibratedClassifierCV` com `ensemble=False`. O calibrador é ajustado
  em previsões fora da amostra e a floresta é treinada uma vez no treino inteiro, então
  continua existindo uma única floresta, com as mesmas árvores da versão sem calibração.
  `sigmoid` ajusta uma curva logística (método de Platt) e é mais seguro com poucos dados;
  `isotonic` é mais flexível, mas pode sobreajustar. As duas mudam o Brier e o log loss.
  O `sigmoid` é uma curva sempre crescente e **não muda o `roc_auc`** (a ordem das
  pacientes continua a mesma). O `isotonic` é uma escada: pacientes com probabilidades
  próximas podem cair no mesmo degrau e empatar, o que **pode mudar o `roc_auc`** (em
  geral, um pouco para baixo). Parâmetro: `n_splits: 5`.
- `threshold: tuned`: em vez de prever 1 quando a probabilidade passa de 0.5, escolhe o
  limiar que maximiza `scoring` (`f1` por padrão, ou `balanced_accuracy`). Não muda o
  `roc_auc`. Não use `scoring: recall`: o limiar ótimo seria 0 (tudo positivo).

### Modelos

| Opção | O que é |
|---|---|
| `random_forest` | `RandomForestClassifier` do scikit-learn (Breiman, 2001). O OOB fica ligado (não muda as previsões) |
| `balanced_rf` | `BalancedRandomForestClassifier` do imbalanced-learn: cada árvore recebe um sorteio com as classes equilibradas por subamostragem (Chen, Liaw e Breiman, 2004). Usa `sampling_strategy: all`, `replacement: true` e `bootstrap: false` |

Hiperparâmetros usados nas rodadas:

| Parâmetro | O que controla | Padrão |
|---|---|---|
| `n_estimators` | número de árvores | 100 |
| `max_features` | variáveis sorteadas por corte (`sqrt`, `log2`, fração ou `null` = todas) | `sqrt` |
| `max_samples` | tamanho do sorteio de pacientes de cada árvore (fração) | `null` (= tamanho do treino) |
| `max_depth` | profundidade máxima de cada árvore | `null` (completa) |
| `min_samples_leaf` | mínimo de pacientes por folha | 1 |
| `class_weight` | `balanced` ou `balanced_subsample` (pesos recalculados em cada bootstrap) | `null` |
| `random_state` | semente da floresta | 42 |

> Com as 8 variáveis originais, `sqrt` sorteia 2 variáveis por corte, `log2` sorteia 3,
> `0.5` sorteia 4 e `null` usa as 8. Com `max_features: null` a Random Forest vira
> **bagging** puro. Com flags ou variáveis derivadas há mais colunas, e esses números sobem.

**Parâmetros iguais ao padrão saem da config.** `{n_estimators: 100, max_features: sqrt}` é
gravado como `{name: random_forest}`. Assim a mesma floresta tem sempre o mesmo hash, escrita
com ou sem os padrões, e o runner não a executa duas vezes (por exemplo,
`R3_tamanho__max_depth=None_min_samples_leaf=1` é o próprio baseline do R3 e é pulado).

### Métodos de imputação

Cinco métodos, um de cada família:

| Opção | Família | Como funciona |
|---|---|---|
| `none` | sem imputação | em cada corte, a floresta decide para que lado mandar os ausentes |
| `constant` | ausência como informação | preenche com -1 (`fill_value`); como todo valor real é maior, um único corte separa quem não fez o exame |
| `median` | estatística simples | mediana da coluna, aprendida no treino |
| `knn_scaled` | vizinhos | média das 5 pacientes mais parecidas (`n_neighbors`), com a distância calculada nas colunas padronizadas |
| `missforest` | modelo | prevê cada coluna ausente com uma Random Forest de 50 árvores (`n_estimators`) treinada nas outras, em 5 rodadas (`max_iter`) (Stekhoven e Bühlmann, 2012) |

O `iterative` (a mesma ideia do `missforest`, mas com regressão linear) fica disponível só
para a comparação opcional "linear x não linear", comentada no `r4_ausentes.yaml`. A
imputação por proximidade de Breiman e a imputação múltipla (MICE) são citadas no artigo,
mas não cabem no pipeline.

Não há etapa de tratamento de outliers nem de padronização: as árvores só usam a ordem dos
valores.

## 5. Decisões do projeto

As justificativas e os achados de cada rodada vão em
[`docs/anotacoes_artigo.md`](docs/anotacoes_artigo.md).

- **Mesmo split e mesma CV do DecisionTree** (70/30 estratificado, seed 42; CV 5×5
  estratificada), para permitir a comparação direta entre os dois modelos.
- **Zeros impossíveis viram ausentes** em Glucose, BloodPressure, SkinThickness, Insulin e BMI.
- **Métrica principal: `roc_auc`.** Acurácia não serve: prever "não diabético" para todos já
  acerta 65,2%. No desbalanceamento (R5), F1 ou `balanced_accuracy`, com recall ≥ 0,70.
- **Métricas de probabilidade:** `brier` (Brier score) e `log_loss` são registradas em todas
  as execuções, para que o R8 possa comparar com as rodadas anteriores. Nas duas, **menor é
  melhor**.
- **Regra de escolha:** entre as configs sem diferença significativa para a melhor
  (`compare.py`), fica a mais simples.
- **Sem SMOTE nem outras amostras sintéticas.**
- **Ordem das rodadas:** primeiro os hiperparâmetros da floresta (R1–R3), depois um
  tratamento de dados por rodada (R4–R6), a combinação (R7) e, por último, a calibração (R8).

## 6. Escrevendo experimentos

O formato do YAML é o mesmo do projeto anterior; a diferença é que o `model` tem `name`:

```yaml
baseline_name: R1_baseline
baseline:
  zeros: to_nan
  missing_flags: none
  imputer: median
  features: none
  selection: none
  balancing: none
  model: {name: random_forest, n_estimators: 100}

experiments:
  - name: R1_arvores
    sweep: {model.n_estimators: [50, 100, 200, 300, 500, 1000]}

  - name: R5_peso_limiar_f1
    override:
      model: {name: random_forest, n_estimators: 300, class_weight: balanced}
      threshold: {name: tuned, scoring: f1}

  - name: R4_ausentes
    override: {imputer: median}
    sweep:
      imputer.name: [none, constant, median, knn_scaled, missforest]
      missing_flags.name: [none, flags]
```

- O `sweep` gera um experimento por valor. A chave é `etapa.parametro`; com mais de uma
  chave, roda todas as combinações. O nome mostra o parâmetro
  (`R1_arvores__n_estimators=300`) ou, quando o sweep troca a opção, a etapa
  (`R4_ausentes__imputer=knn_scaled_missing_flags=flags`).
- Configs já executadas (mesmo hash) são puladas.
- O `override` **substitui a etapa inteira**, inclusive o `model`: para mudar só
  `class_weight`, repita `n_estimators` e os demais vencedores.
- `threshold: none` ou `calibration: none` num override tira essa etapa (usado nas ablações
  do R7).
- O baseline roda automaticamente com o nome de `baseline_name`. Do R2 em diante ele é igual
  ao vencedor da rodada anterior e é pulado pelo hash (continua com o nome que recebeu
  antes). Por isso o `compare.py --rodada` precisa de `--incluir <vencedor anterior>`.
- Do R2 em diante, os valores marcados com `# ATUALIZAR: vencedor de Rx` são provisórios:
  **não rode uma rodada antes de atualizar o baseline dela**.
- Antes de rodar qualquer coisa, o script valida todos os experimentos (nomes de opções,
  hiperparâmetros do modelo, combinações proibidas). Um erro de digitação para tudo com uma
  mensagem clara.

## 7. Rodadas

Cada rodada responde a uma pergunta e muda uma coisa só. O vencedor de uma vira o baseline
da seguinte. Cada YAML traz no cabeçalho a pergunta, o critério e o comando para fechar a
rodada.

| Rodada | Pergunta | O que varia | Critério |
|---|---|---|---|
| **R1** `r1_num_arvores` | Quantas árvores bastam para o resultado estabilizar? | `n_estimators`: 50, 100, 200, 300, 500, 1000 | `roc_auc`; o **menor** número sem diferença significativa para o melhor |
| **R2** `r2_max_features` | Quanto vale a pena decorrelacionar as árvores? | `max_features`: `sqrt`, `log2`, 0.5, `null`; `max_samples`: 0.5, 0.7 | `roc_auc`; no empate, o `sqrt` (padrão da literatura) |
| **R3** `r3_tamanho_arvores` | Árvores completas ou limitadas? | `max_depth`: null, 4, 6, 8, 12 × `min_samples_leaf`: 1, 3, 5, 10 | `roc_auc`; a mais rasa ou com folhas maiores sem diferença significativa |
| **R4** `r4_ausentes` | Como tratar os exames não medidos? | 5 imputadores × `missing_flags` (com/sem) = 10 execuções | `roc_auc`; a mais simples (`none` → `constant` → `median` → `knn_scaled` → `missforest`) |
| **R5** `r5_desbalanceamento` | Como encontrar mais diabéticas sem SMOTE? | `class_weight` (`balanced`, `balanced_subsample`), `balanced_rf`, `threshold: tuned`, `random_under` | F1 ou `balanced_accuracy`, com recall ≥ 0,70 |
| **R6** `r6_variaveis` | HOMA-IR, razões e faixas ainda ajudam? | `features`: `homa`, `ratios`, `bmi_age_bins` | `roc_auc`; sem variável nova, se empatar |
| **R7** `r7_combinacao` | Os vencedores funcionam juntos? | todos os vencedores + ablação (tirar um de cada vez) | o critério da rodada que decidiu cada ponto |
| **R8** `r8_calibracao` | As probabilidades podem ser lidas como risco? | `calibration`: `sigmoid`, `isotonic` (opcional: as candidatas empatadas do R5, já calibradas) | `brier`; a mais simples sem diferença significativa (nas candidatas do R5, o critério do R5) |

Depois do R8:

| Rodada | O que é |
|---|---|
| **R9** Escolha | registrar a config final e a justificativa de cada rodada em `docs/anotacoes_artigo.md` |
| **R10** Teste final | `final_test.py` no vencedor, **uma única vez** (inclui Brier e curva de calibração) |
| **R11** DT × RF + interpretabilidade | `dt_vs_rf.py` (CV fold a fold + teste lado a lado) e `explain.py` (PDP, SHAP e árvore substituta) |
| **R12** Robustez e estabilidade | `r12_robustez.yaml` (o vencedor com 5 outras sementes) e `stability.py` (as importâncias mudam entre folds e sementes?) |
| **Complementar** OOB × CV | `oob_vs_cv.py`: o erro OOB concorda com a validação cruzada em todas as execuções? |

### Hipóteses a discutir no artigo

- **R1:** o AUC forma um platô. Mais árvores não causam overfitting, só custam tempo
  (Probst e Boulesteix, 2018).
- **R2:** se `null` (bagging) empatar com `sqrt`, o ganho vem do sorteio de pacientes, e não
  do sorteio de variáveis.
- **R3:** contraste com o DT, em que a árvore precisava ser rasa. Na floresta, árvores
  completas devem funcionar bem.
- **R4:** duas perguntas. (1) A ausência informa? (`none` / `constant` / `flags` contra os
  imputadores.) (2) Vale imputar com sofisticação? (`median` → `knn_scaled` → `missforest`.)
- **R6:** o ganho das variáveis derivadas deve ser menor que no DT, porque muitas árvores
  conseguem aproximar interações como glicose × insulina.
- **R8:** as probabilidades da floresta tendem a ficar afastadas de 0 e de 1
  (Niculescu-Mizil e Caruana, 2005), e os pesos de classe do R5 as deslocam para a classe
  positiva. A calibração deve melhorar o Brier sem mudar o AUC (exatamente igual no
  `sigmoid`; quase igual no `isotonic`): **AUC alto não garante probabilidade confiável**.
  Se o R5 escolheu `class_weight`, boa parte da melhora no Brier vem de desfazer o
  deslocamento proposital das probabilidades: o peso aumenta o recall, a calibração
  devolve o risco ao lugar. Como o R5 foi decidido com probabilidades sem calibração, o
  `r8_calibracao.yaml` traz um bloco opcional para conferir as candidatas empatadas do R5
  já calibradas.
- **R11:** a árvore substituta mostra quanto da floresta dá para resumir em regras legíveis.
  Se ela tiver alta fidelidade com poucas folhas, boa parte da interpretabilidade do DT pode
  ser recuperada.
- **R12:** as importâncias da floresta devem ser bem mais estáveis que as da árvore única,
  que muda de estrutura com pequenas mudanças nos dados.
- **OOB × CV:** se os dois concordarem, o OOB é um atalho confiável para avaliar florestas
  sem validação cruzada, como propôs Breiman (2001).

## 8. Onde ficam os resultados

Cada execução cria uma pasta nova em `results/runs/`:

```
results/runs/20261105_142558_0e1f8f_R1_arvores__n_estimators=300/
├── config.yaml                   config completa usada (padrões do modelo omitidos)
├── metrics.json                  média e desvio de cada métrica na validação cruzada
├── fold_scores.csv               as 25 notas individuais (5 folds x 5 repetições)
├── meta.json                     data, hash, versões, hash do train.csv, limiar (se houver)
├── pipeline.joblib               o pipeline treinado (tratamentos + floresta + envoltórios)
├── oob.json                      roc_auc e brier out-of-bag da floresta final (ou o motivo de não haver)
├── forest_stats.json             nº de árvores; profundidade e nº de folhas: média, mín. e máx.
├── importances_mdi.csv           importância por impureza da floresta final
├── importances_permutation.csv   importância por permutação: média e desvio nos 25 folds
├── fold_importances.csv          as duas importâncias em cada um dos 25 folds
├── importances.png               as duas importâncias lado a lado
└── tree_example.png              uma árvore da floresta, desenhada até a profundidade 3
```

Como no projeto anterior, a validação cruzada treina 25 florestas só para medir o
desempenho, e depois uma **floresta final** é treinada com o `train.csv` inteiro. É ela que
fica salva.

**As duas importâncias:**

- **Impureza (MDI):** a padrão do scikit-learn, calculada nas colunas que chegam à floresta
  (depois dos tratamentos, então inclui flags e HOMA, se houver). Tende a favorecer variáveis
  contínuas com muitos valores distintos (Strobl et al., 2007).
- **Permutação:** em cada fold, embaralha um exame **original** no fold de validação e mede
  quanto o `roc_auc` cai (5 repetições por exame). Não tem o viés da MDI e é medida em dados
  que a floresta não viu. Diferenças entre as duas rendem discussão no artigo (uma variável
  contínua pode aparecer bem na MDI e quase não importar na permutação).

**OOB:** existe só com bootstrap e sem reamostragem antes da floresta. No `balanced_rf` e com
`random_under`, o `oob.json` registra o motivo de não haver. Numa execução calibrada, o OOB é
o da floresta sem calibração.

Uma floresta não tem regras legíveis como o `tree.txt` do projeto anterior. A
`tree_example.png` serve só para ilustrar como é uma das árvores; ela sozinha **não**
representa o modelo.

Para usar o pipeline num notebook:

```python
import joblib
import src                      # aplica a configuração do projeto antes de carregar
from src.builder import forest_of
from src.data import load_train

pipe = joblib.load("results/runs/<pasta>/pipeline.joblib")
floresta = forest_of(pipe)       # o RandomForestClassifier, mesmo sob calibração/limiar
X, y = load_train()
pipe.predict_proba(X.head())[:, 1]
```

### Tabela comparativa

`python summarize.py` lê todas as pastas de `results/runs/` e gera `results/summary.csv`,
ordenado pela métrica escolhida (`--metric`; no `brier` e no `log_loss`, do menor para o
maior). Além das métricas, traz o `roc_auc` OOB, o nº de árvores, a profundidade média, o
limiar e o tempo de cada execução. A coluna `pipeline` resume a config, por exemplo
`to_nan > knn_scaled | random_forest(n_estimators=300,class_weight=balanced) > tuned(scoring=f1)`.

Essa tabela é sempre derivada das pastas: se algo der errado, apague-a e gere de novo.

### Garantias contra sobrescrita

- O runner só **cria** pastas. Ele nunca apaga nem edita nada em `results/runs/`.
- Se a mesma config, com os mesmos dados, já foi executada, ele avisa e pula. Com
  `--force`, cria uma pasta nova ao lado da antiga.
- Enquanto roda, a execução fica numa pasta `.tmp_...`. Ela só ganha o nome definitivo
  quando tudo foi gravado, então uma execução interrompida nunca conta como feita.
- Se você mudar o **código** de um método já existente (não a config), o hash não muda e o
  runner vai pular. Nesse caso, use `--force` (ou crie uma opção nova).

## 9. Comparando execuções

O `compare.py` é o mesmo do projeto anterior: compara as 25 notas fold a fold com o **teste
t corrigido de Nadeau e Bengio (2003)**, com α = 0,05.

```bash
python compare.py <run_A> <run_B>                                    # duas execuções
python compare.py --rodada R2_ --incluir <vencedor do R1> --salvar   # fecha uma rodada
python compare.py --rodada R5_ --metric f1 --incluir <vencedor do R4>
python compare.py --rodada R8_ --metric brier --incluir <vencedor do R7>
```

- No modo `--rodada`, a referência é a execução de melhor média (a maior; no `brier` e no
  `log_loss`, a menor), ou a passada em `--ref`. A tabela mostra também o **nº de árvores**,
  a **profundidade média** e o **nº médio de folhas** de cada floresta.
- **Árvore × floresta:** como o split e os folds são os mesmos, o `compare.py` aceita a pasta
  de uma execução do outro projeto. Execuções do DecisionTree não têm `brier`/`log_loss`;
  essas métricas ficam de fora da comparação.

```bash
python compare.py ../DecisionTree/results/runs/<vencedor_DT> <vencedor_RF>
```

p ≥ α quer dizer "sem diferença significativa", não "equivalentes". Cada rodada faz várias
comparações sem correção para comparações múltiplas; p-valores perto de α devem ser lidos
com cautela.

## 10. Teste final

Só o `final_test.py` lê o `test.csv` durante o projeto. Ele carrega o `pipeline.joblib` da
execução indicada, avalia uma vez no teste e grava `metrics.json` (com Brier e log loss),
`confusion_matrix.csv`, `calibration_curve.png` e `meta.json` em `results/final/<run>/`. Se a
execução já foi avaliada, ele apenas mostra o resultado guardado.

Os scripts do R11 (`dt_vs_rf.py` e `explain.py`) também usam o `test.csv`, mas só rodam
depois do teste final daquela execução (eles conferem isso) e não mudam nenhuma escolha.

## 11. Análises do vencedor

### Árvore × floresta (`dt_vs_rf.py`, R11)

```bash
python dt_vs_rf.py --rf <vencedor RF> --dt 20261024_145908_a1b1c4_R3_peso_balanceado
```

Exige o teste final feito nos dois projetos. Grava em `results/dt_vs_rf/<run>/`:

| Arquivo | O que mostra |
|---|---|
| `cv.csv` | as métricas da CV lado a lado, com o teste t corrigido fold a fold |
| `teste.csv` | as métricas no teste dos dois modelos salvos, com as mesmas funções (inclui o Brier da árvore, que o DecisionTree não registrava) |
| `confusao.csv` | as duas matrizes de confusão |
| `roc_calibracao.png` | curvas ROC e de calibração dos dois no teste |

O script confere se a árvore reavaliada bate com o teste guardado no DecisionTree.

### Interpretabilidade (`explain.py`, R11)

```bash
python explain.py --run <vencedor> [--n-background 50] [--surrogate-depth 3]
```

Grava em `results/explain/<run>/`:

| Arquivo | O que mostra |
|---|---|
| `pdp.png`, `pdp.csv` | **dependência parcial** (Friedman, 2001): o risco médio previsto quando cada exame varia do percentil 5 ao 95, com as outras variáveis como estão. Ex.: a partir de que glicose o risco dispara |
| `shap_summary.png` | **SHAP**: quanto cada exame empurrou o risco de cada paciente do teste para cima ou para baixo (cor = valor do exame; x = exame não medido) |
| `shap_values.csv` | os valores SHAP de cada paciente do teste |
| `shap_exemplos.csv` | 8 pacientes explicadas (2 de cada: acertos e erros confiantes), por exemplo: "Glucose=160 (+0.30); BMI=38 (+0.08); Insulin=ausente (+0.05)" |
| `surrogate_tree.txt` / `.png` | **árvore substituta**: uma árvore rasa treinada para imitar as previsões da floresta (não o `Outcome`) |
| `surrogate_fidelity.csv` | fidelidade da árvore substituta (% de pacientes em que ela concorda com a floresta), no treino e no teste, para profundidades de 2 a 6 |

**Como o SHAP é calculado:** sem a biblioteca `shap`. Com 8 exames há só 2⁸ = 256
coalizões, então o script calcula os **valores de Shapley exatos** na versão intervencional
(a mesma do KernelSHAP; Lundberg e Lee, 2017), usando 50 pacientes do treino como fundo. Eles
explicam o modelo inteiro (tratamentos + floresta + calibração) em termos dos 8 exames
originais, e a soma dos valores de cada paciente é exatamente o risco dela menos o risco
médio do fundo (o `meta.json` registra o erro dessa conferência).

A árvore substituta liga os dois trabalhos: as regras dela podem ser comparadas com as da
árvore do DecisionTree. Se forem parecidas, a floresta "pensa" como a árvore, só que com
mais estabilidade.

### Estabilidade das importâncias (`stability.py`, R12)

```bash
python run_experiments.py experiments/r12_robustez.yaml
python stability.py --run <vencedor> --run-dt <vencedor do DT> --seeds R12_
```

Usa o `fold_importances.csv` do vencedor e retreina o vencedor do DT nos mesmos 25 folds.
Para cada modelo e tipo de importância, mede:

- a **variação** de cada variável entre os folds (desvio e coeficiente de variação);
- a **concordância do ranking** das variáveis entre os folds (correlação de Spearman média
  e mínima entre os 300 pares de folds);
- em quantos folds a variável mais importante é a mesma;
- com `--seeds`: o `roc_auc` de cada semente e a concordância do ranking entre as florestas
  finais das sementes.

Grava em `results/stability/<run>/` as tabelas e um boxplot da importância por fold, RF e DT
lado a lado.

### OOB × validação cruzada (`oob_vs_cv.py`)

Lê o `oob.json` e o `metrics.json` de todas as execuções e grava `results/oob_vs_cv.csv` e
`results/oob_vs_cv.png` (AUC e Brier OOB × CV), com a diferença média e as correlações.
Ficam de fora da conta as execuções sem OOB e as calibradas.

Limitações a registrar no artigo:

- o OOB só existe com bootstrap; no `balanced_rf` e com `random_under`, as amostras "de
  fora" seguem outra lógica;
- os tratamentos do pipeline (imputação, flags) são ajustados no treino inteiro antes da
  floresta, então as pacientes OOB já influenciaram a imputação. O OOB fica ligeiramente
  otimista em relação a isso; por outro lado, cada paciente OOB é avaliada por só ~1/3 das
  árvores, o que o deixa um pouco pessimista.

## 12. Criando um método novo

Exemplo: um imputador que preenche com a moda.

1. Em `src/steps/imputers.py`:
   ```python
   def most_frequent():
       return SimpleImputer(strategy="most_frequent")
   ```
2. Em `src/registry.py`, na entrada `"imputer"`:
   ```python
   "most_frequent": imputers.most_frequent,
   ```
3. No YAML:
   ```yaml
   - name: R4_moda
     override: {imputer: most_frequent}
   ```

Transformadores que aprendem algo com os dados devem calcular esses valores no `fit` e só
aplicá-los no `transform`. É isso que impede a contaminação com os dados de validação.

## Referências

- Breiman, L. *Bagging Predictors*. Machine Learning, 24, 123–140, 1996.
- Breiman, L. *Random Forests*. Machine Learning, 45, 5–32, 2001.
- Brier, G. W. *Verification of forecasts expressed in terms of probability*. Monthly
  Weather Review, 78(1), 1–3, 1950.
- Chen, C.; Liaw, A.; Breiman, L. *Using Random Forest to Learn Imbalanced Data*.
  Technical Report 666, University of California, Berkeley, 2004.
- Friedman, J. H. *Greedy function approximation: a gradient boosting machine*. Annals of
  Statistics, 29(5), 1189–1232, 2001.
- Lundberg, S. M.; Lee, S.-I. *A unified approach to interpreting model predictions*.
  Advances in Neural Information Processing Systems 30 (NeurIPS), 2017.
- Nadeau, C.; Bengio, Y. *Inference for the Generalization Error*. Machine Learning, 52,
  239–281, 2003.
- Niculescu-Mizil, A.; Caruana, R. *Predicting good probabilities with supervised
  learning*. Proceedings of the 22nd International Conference on Machine Learning (ICML),
  2005.
- Probst, P.; Boulesteix, A.-L. *To tune or not to tune the number of trees in random
  forest*. Journal of Machine Learning Research, 18, 2018.
- Probst, P.; Wright, M. N.; Boulesteix, A.-L. *Hyperparameters and tuning strategies for
  random forest*. WIREs Data Mining and Knowledge Discovery, 9(3), 2019.
- Stekhoven, D. J.; Bühlmann, P. *MissForest — non-parametric missing value imputation for
  mixed-type data*. Bioinformatics, 28(1), 112–118, 2012.
- Strobl, C.; Boulesteix, A.-L.; Zeileis, A.; Hothorn, T. *Bias in random forest variable
  importance measures: Illustrations, sources and a solution*. BMC Bioinformatics, 8:25, 2007.
