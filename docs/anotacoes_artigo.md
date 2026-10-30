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
- Vencedor: 20261029_113816_5ffbd2_R5_limiar_f1 (sem peso de classe, sem calibração,
  limiar ajustado para F1, ~0,39). Substitui o vencedor do R5 (class_weight=balanced).

- Brier antes e depois; o AUC ficou igual?
  - Antes (class_weight=balanced, sem calibração): Brier 0,172, ROC-AUC 0,826.
  - Sigmoide: Brier 0,160 (melhor, p = 0,001 contra sem calibração), ROC-AUC 0,826
    -> idêntico, como esperado (a curva é crescente e não muda a ordem das pacientes).
  - Isotônica: Brier 0,163 (empata com o sigmoide, p = 0,193), ROC-AUC 0,821 -> caiu 0,005,
    pelos empates criados pelos degraus.

- Hipótese (peso de classe + calibração): confirmada em parte.
  - Confirmado: a calibração melhora o Brier por desfazer o deslocamento das probabilidades
    para a classe positiva causado pelo class_weight.
  - Refutado: o recall NÃO se manteve. Com o corte em 0,5, as probabilidades calibradas
    (mais baixas) fazem o modelo apontar menos diabéticas: recall 0,725 -> 0,599 (sigmoide)
    e 0,576 (isotônica), abaixo do critério de 0,70 do R5.
  - Solução: separar risco e decisão, calibrando as probabilidades e ajustando o limiar.

- Candidatas empatadas do R5 conferidas já calibradas? A escolha se manteve?
  - Testadas: R8_cand_limiar (sigmoide + limiar, sem peso) e R8_cand_peso_limiar
    (sigmoide + limiar + peso), junto com o R5_limiar_f1 (só limiar).
  - Brier: as três empatam com a melhor (R8_cand_limiar 0,159; R8_cand_peso_limiar 0,160,
    p = 0,885; R5_limiar_f1 0,162, p = 0,303).
  - F1 (--ref R5_limiar_f1, 0,654): empata com R5 peso 0,666 (p = 0,455),
    R8_cand_peso_limiar 0,658 (p = 0,812) e R8_cand_limiar 0,653 (p = 0,948).
  - Recall: R5_limiar_f1 0,726; R8_cand_limiar 0,718; R8_cand_peso_limiar 0,744.
  - A escolha do R5 NÃO se manteve: o class_weight=balanced sai, porque piora o Brier
    (0,172) sem ganho significativo de F1. Fica a opção mais simples que passa nos três
    critérios: R5_limiar_f1 (sem peso, sem calibração).
  - Conclusão: o problema das probabilidades vinha do peso de classe. Sem ele, a floresta
    já sai quase calibrada (Brier 0,162, empatado com o sigmoide), e o recall é recuperado
    baixando o limiar de decisão, em vez de distorcer o treino.
  - Atenção: o limiar ajustado deixa o recall mais instável entre folds
    (dp 0,109, contra 0,072 com o peso de classe).

### Acurácia do modelo final (validação cruzada)
- Acurácia: 0,734 ± 0,042. Modelo trivial ("não diabética" para todas): 0,651.
- Fica entre as mais baixas das 54 execuções; a maior é 0,772 (R3, profundidade 6 /
  folha 10), mas com recall de só 0,550.
- Motivo: ao baixar o limiar de 0,5 para ~0,39, o modelo encontra mais diabéticas
  (recall 0,726 contra 0,550) à custa de mais falsos alarmes nas não diabéticas. Num grupo
  de 231 pacientes (~81 diabéticas), isso equivale a ~14 diabéticas a mais encontradas e
  ~23 falsos alarmes a mais, ou ~9 erros a mais no saldo.
- A acurácia trata os dois erros como iguais; em rastreamento, deixar passar uma diabética
  é mais grave. Por isso ela não foi usada como critério de seleção.
- Na balanced accuracy, que pondera igualmente as duas classes, o modelo final é superior
  (0,732 contra 0,721 do modelo de maior acurácia).

## R9 – Escolha final

### Configuração final
Pasta: results\runs\20261029_113816_5ffbd2_R5_limiar_f1

| Etapa              | Escolha                                          | Rodada |
|--------------------|--------------------------------------------------|--------|
| Zeros              | tratados como ausentes (NaN), exceto gestações    | R4     |
| Imputação          | mediana, calculada só no treino de cada fold      | R4     |
| Flags de ausência  | não                                              | R4     |
| Variáveis novas    | não                                              | R6     |
| Nº de árvores      | 100                                              | R1     |
| max_features       | sqrt                                             | R2     |
| max_depth          | 4                                                | R3     |
| min_samples_leaf   | 10                                               | R3     |
| Desbalanceamento   | limiar de decisão ajustado para F1 (~0,39)        | R5/R8  |
| Peso de classe     | não                                              | R8     |
| Calibração         | não                                              | R8     |

Desempenho na validação cruzada (5 folds x 5 repetições):
ROC-AUC 0,827 · recall 0,726 · F1 0,654 · balanced accuracy 0,732 · Brier 0,162 ·
acurácia 0,734.

### Regra geral de decisão
Em cada rodada, as configurações foram comparadas com o teste t corrigido de
Nadeau-Bengio (α = 0,05). Entre as que não diferiam significativamente da melhor, foi
escolhida a mais simples. Exceção: no número de árvores (R1), "mais simples" não se
aplica, porque mais árvores não causam sobreajuste.

### Justificativa por rodada

**R1 – Número de árvores: 100.**
As 6 opções (50 a 1000) empataram (ROC-AUC 0,815 a 0,823). Não se escolheu 50 porque o
número de árvores não controla sobreajuste (Breiman, 2001) e deve ser "grande o
suficiente" (Probst e Boulesteix, 2018); o ganho fica pequeno a partir de 64–128 árvores
(Oshiro et al., 2012); com 50, a parcela (1 − ρ)σ²/B da variância (Eq. 1) é o dobro da de
100, o que adicionaria ruído às rodadas seguintes; 50 teve a menor média em todas as
métricas; e o custo de 100 é pequeno. A execução foi reproduzível entre duas máquinas.

**R2 – max_features: sqrt (padrão).**
[CONFERIR: faixa de ROC-AUC e p da R2 em results\comparacoes\R2_roc_auc.csv.]
As opções testadas (sqrt, log2, 0,5, None e max_samples 0,5/0,7) não superaram
significativamente o padrão, que foi mantido por ser o mais simples.

**R3 – Tamanho das árvores: max_depth 4, min_samples_leaf 10.**
As 20 combinações empataram (ROC-AUC 0,819 a 0,832). Escolhida a mais simples nos dois
eixos: 0,827, p = 0,304 contra a melhor, ~12 folhas por árvore contra ~93 das árvores
completas. As árvores completas, padrão do RF, ficaram em penúltimo (0,819, p = 0,080),
na mesma direção do DecisionTree (árvore rasa), mas de forma mais suave.

**R4 – Valores ausentes: mediana, sem flags.**
As 10 opções empataram (ROC-AUC 0,819 a 0,833; mediana sem flags 0,827, p = 0,445).
Imputadores mais complexos (KNN, missforest) e as flags não trouxeram ganho; o missforest
ainda custou ~270 s por execução. A mediana é a opção mais simples e a mesma do
DecisionTree.

**R5 – Desbalanceamento (escolha revista na R8).**
Sem tratamento, o recall fica em ~0,54 e o F1 em 0,605, significativamente pior
(p = 0,032). As 7 técnicas testadas empataram no F1 (0,652 a 0,675) e passaram no filtro
de recall ≥ 0,70. A escolha inicial foi class_weight=balanced (F1 0,666, recall 0,725),
depois substituída na R8 pelo ajuste de limiar.

**R6 – Variáveis derivadas: nenhuma.**
HOMA-IR, razões e faixas empataram em ROC-AUC (0,826 a 0,832; sem variável nova
p = 0,348) e em F1 (0,664 a 0,682; p = 0,310). Coerente com a hipótese: a floresta
aproxima sozinha interações como glicose x insulina.

**R7 – Ablação: todos os elementos mantidos.**
- Sem peso de classe: F1 significativamente pior (0,605, p = 0,043), recall ~0,54.
- Árvores completas: ROC-AUC e F1 empatam, mas o recall cai para 0,670 (< 0,70) e as
  árvores passam de 12 para 94 folhas; a poda fica, por ser também a opção mais simples.
- Flags, imputação, variáveis, subamostragem e limiar já estavam no padrão.

**R8 – Calibração: revisão da escolha do R5.** 
- Com class_weight=balanced, o Brier era 0,172. A calibração sigmoide reduziu para 0,160
(p = 0,001), mas derrubou o recall para 0,599, porque desfaz a inflação das
probabilidades e o corte continua em 0,5. Testando as candidatas do R5 com calibração e
limiar, três passaram em todos os critérios (recall ≥ 0,70, F1 e Brier empatados com o
melhor). Escolhida a mais simples: limiar F1 sem peso e sem calibração (Brier 0,162,
p = 0,303 contra o melhor; F1 0,654, empatado com todas, p ≥ 0,455). Conclusão: o peso
de classe distorcia as probabilidades; sem ele, a floresta já sai quase calibrada, e o
recall é recuperado baixando o limiar.

### Limitações registradas
- O recall do limiar ajustado é mais instável entre folds (dp 0,109).
- A acurácia (0,734) é menor que a de modelos com corte em 0,5 (até 0,772), troca
  consciente por um recall maior.
- Muitas rodadas terminaram em empate: com ~537 pacientes no treino, o teste tem pouco
  poder para detectar diferenças pequenas ("não significativo" ≠ "igual").
- 4 execuções da R5 foram refeitas porque a primeira rodada usou o model antigo nos
  overrides; as pastas erradas estão em results\descartados.
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
