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

- Vencedor: 20261028_230418_091b13_R1_baseline (n_estimators = 100)
- Observações: as 6 configurações (50 a 1000) empataram pelo teste t corrigido
  (ROC-AUC de 0,815 a 0,823; p ≥ 0,14 contra a de 1000). Platô a partir de ~200–300.
  Não foi escolhida a de 50: mais árvores não sobreajustam (Breiman 2001), o número de
  árvores deve ser "grande o suficiente" (Probst e Boulesteix 2018), o ganho  fica pequeno
  a partir de 64–128 (Oshiro et al. 2012), e com 50 a variância da floresta (Eq. 1) é o
  dobro da de 100, o que adiciona ruído às rodadas seguintes. 100 = padrão do sklearn.

### R2 – max_features
- Vencedor: 20261028_230418_091b13_R1_baseline (sqrt, sem max_samples, 100 árvores)
- O bagging (`max_features: null`) empatou com o `sqrt`? Sim: 0,813 x 0,819, p = 0,140.
  Todas as 6 empataram (ROC-AUC 0,813 a 0,823); no empate fica o sqrt (critério da rodada).
  max_samples 0,7 teve a maior média (0,823, p = 0,429 contra o sqrt), mas não foi significativo.

### R3 – Tamanho das árvores
### R3 – Tamanho das árvores
- Vencedor: 20261028_235130_c53880_R3_tamanho__max_depth=4_min_samples_leaf=10
- Todas as 20 empataram (ROC-AUC 0,819 a 0,832; ref. max_depth 6 / leaf 5). Escolhida a mais
  simples nos dois eixos: 0,827, p = 0,304, ~12 folhas por árvore (contra 93 das completas).
- Contraste com o DT: as árvores completas (padrão do RF) ficaram em penúltimo (0,819,
  p = 0,080). Mesma direção do DT (árvore rasa), mas bem mais suave na floresta.

### R4 – Valores ausentes
### R4 – Valores ausentes
- Vencedor: mediana sem flags (mantém 20261028_235130_c53880_R3_tamanho__max_depth=4_min_samples_leaf=10)
- Todas as 10 empataram (ROC-AUC 0,819 a 0,833; ref. KNN com flags). Mediana sem flags: 0,827, p = 0,445.
- Imputadores mais complexos (KNN, missforest) e as flags de ausência não trouxeram ganho
  significativo; missforest ainda custou ~270 s por execução. Escolhida a opção mais simples,
  igual ao pré-processamento do DecisionTree.




### R5 – Desbalanceamento
- Vencedor: 20261029_133225_eae571_R5_pesos__class_weight=balanced
- Critério: F1 entre as configs com recall >= 0,70. As 7 técnicas empataram (F1 0,652 a 0,675;
  ref. subamostragem) e todas passaram no filtro de recall.
- Sem tratamento, o F1 cai para 0,605 (p = 0,032, única diferença significativa) e o recall
  fica em ~0,54: tratar o desbalanceamento melhora de fato.
- Escolhido class_weight=balanced (F1 0,666, recall 0,725): um único parâmetro, usa todos os
  dados e é a mesma técnica do DecisionTree.
- Obs.: 4 execuções foram refeitas porque a primeira rodada usou o model antigo nos overrides
  (300 árvores, sem poda); as pastas erradas estão em results\descartados.
     - Balanced accuracy confirma: as 7 empatam (0,732 a 0,750); class_weight=balanced 0,742, p = 0,556.
     O baseline (0,709, p = 0,099) não difere na bal_acc, mas já está fora pelo recall (~0,54).

### R6 – Variáveis derivadas
- Vencedor: sem variável nova (mantém 20261029_133225_eae571_R5_pesos__class_weight=balanced)
  - ROC-AUC: empate (0,826 a 0,832; ref. ratios; sem variável nova p = 0,348).
  - F1: empate (0,664 a 0,682; ref. homa; sem variável nova p = 0,310).
- O ganho foi menor que no DT? Nos dois modelos as variáveis derivadas não foram adotadas:
  o DT final também ficou sem variáveis novas. Na floresta, o maior ganho foi de +0,006 de
  ROC-AUC (ratios) e +0,016 de F1 (homa), nenhum significativo. Coerente com a hipótese:
  com muitas árvores combinando cortes, a floresta aproxima sozinha interações como
  glicose x insulina. 


### R7 – Combinação e ablação
- Combinação final = 20261029_133225_eae571_R5_pesos__class_weight=balanced
  (ROC-AUC 0,826, F1 0,666, recall 0,725). Comparações com --ref nessa pasta.
- Sem peso de classe: ROC-AUC empata (0,827, p = 0,760), mas o F1 piora significativamente
  (0,605, p = 0,043) e o recall cai para ~0,54 -> o peso fica.
- Árvores completas: ROC-AUC (0,820, p = 0,515) e F1 (0,651, p = 0,402) empatam, mas o recall
  cai para 0,670 (< 0,70) e as árvores passam de 12 para 94 folhas -> a poda fica
  (é também a opção mais simples).
- Flags, imputação, variáveis, subamostragem e limiar já estavam no padrão: nada a remover.

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
