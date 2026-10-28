# Anotações para o artigo (Random Forest)

Registro das decisões, dos achados e das limitações de cada rodada, para escrever o artigo.
Preencha ao fechar cada rodada: a pasta do vencedor, os números que justificam a escolha e o
que chamou a atenção. Os números vêm do `compare.py --rodada ... --salvar`
(`results/comparacoes/`) e do `summary.csv`.

## 1. Decisões de método (fixas desde o início)

- **Mesmo split e mesma CV do DecisionTree** (70/30 estratificado, seed 42; CV 5×5
  estratificada). O `train.csv` foi copiado do projeto anterior (mesmo sha256), o que permite
  comparar a melhor árvore e a melhor floresta fold a fold.
- **Zeros impossíveis viram ausentes** em Glucose, BloodPressure, SkinThickness, Insulin e BMI.
- **Métrica principal: `roc_auc`**; no desbalanceamento (R5), F1 ou `balanced_accuracy` com
  recall ≥ 0,70; na calibração (R8), Brier.
- **Regra de escolha:** entre as configs sem diferença significativa para a melhor (teste t
  corrigido de Nadeau e Bengio, α = 0,05, sem correção para comparações múltiplas), fica a
  mais simples.
- **Sem SMOTE**, sem tratamento de outliers e sem padronização.
- **Importâncias:** MDI da floresta final e permutação (queda do `roc_auc`) medida nos folds de
  validação, sobre os 8 exames originais.

## 2. Rodadas

Para cada rodada: vencedor (pasta), referência, candidatas empatadas, escolha e justificativa.

### R1 – Número de árvores
- Vencedor:
- Observações (o platô apareceu? a partir de quantas árvores?):

### R2 – max_features
- Vencedor:
- O bagging (`max_features: null`) empatou com o `sqrt`?

### R3 – Tamanho das árvores
- Vencedor:
- Contraste com o DT (lá a árvore precisou ser rasa):

### R4 – Valores ausentes
- Vencedor:
- A ausência informa? (`none` / `constant` / `flags` x imputadores)
- Vale imputar com sofisticação? (`median` → `knn_scaled` → `missforest`)

### R5 – Desbalanceamento
- Vencedor:
- Recall, F1 e balanced_accuracy das candidatas com recall ≥ 0,70:

### R6 – Variáveis derivadas
- Vencedor:
- O ganho foi menor que no DT?

### R7 – Combinação e ablação
- Vencedor:
- Elementos removidos por não fazerem diferença:

### R8 – Calibração
- Vencedor:
- Brier antes e depois; o AUC ficou igual? (deve ser idêntico no sigmoid; no isotonic pode
  cair um pouco, por causa dos empates criados pela escada)
- Hipótese: se o R5 escolheu `class_weight`, a calibração melhora o Brier sobretudo por
  desfazer o deslocamento das probabilidades para a classe positiva, sem perder o recall.
- Candidatas empatadas do R5 conferidas já calibradas (bloco opcional do YAML)? A escolha se
  manteve?

### R9 – Escolha final
- Config final:
- Justificativa:

### R10 – Teste final
- Métricas no teste (CV × teste):
- Matriz de confusão, em termos clínicos:

### R11 – Árvore × floresta e interpretabilidade
- CV fold a fold (`dt_vs_rf.py`):
- Teste lado a lado:
- PDP / SHAP: o que o modelo aprendeu de cada exame?
- Árvore substituta: profundidade, fidelidade e semelhança com as regras do DT:

### R12 – Robustez e estabilidade
- roc_auc entre sementes (desvio, amplitude):
- Concordância do ranking das importâncias (RF × DT):

### Complementar – OOB × CV
- Diferença média e correlação:

## 3. Limitações

- Base pequena (537 pacientes no treino, 231 no teste, 81 diabéticas no teste): grande
  incerteza nas métricas de teste.
- Ordem das rodadas: o desbalanceamento (R5) foi decidido com probabilidades sem calibração;
  a calibração (R8) veio depois e reajusta o limiar. Se não foram conferidas as candidatas
  empatadas do R5 já calibradas, registrar que a escolha do R5 poderia ter sido outra.
- Muitas comparações por rodada sem correção para comparações múltiplas; p-valores perto de α
  merecem cautela.
- O OOB fica ligeiramente otimista em relação à imputação (ajustada no treino inteiro antes
  da floresta) e não existe no `balanced_rf` nem com `random_under`.
- SHAP intervencional: ao trocar um exame pelo valor de outra paciente, pode criar
  combinações pouco realistas quando os exames são correlacionados (ex.: glicose e insulina).
- A importância MDI é enviesada para variáveis contínuas (Strobl et al., 2007); a permutação
  é a referência para conclusões.
