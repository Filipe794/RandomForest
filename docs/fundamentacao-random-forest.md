# Fundamentação teórica: Random Forest

Este documento reúne a base teórica do trabalho. Ele cobre o funcionamento do algoritmo, os hiperparâmetros mais relevantes e os tratamentos de dados indicados para usar com ele.

## 1. Como funciona

O Random Forest (Breiman, 2001) é um **ensemble de árvores de decisão** que combina duas fontes de aleatoriedade para gerar árvores diferentes entre si:

1. **Bagging (bootstrap aggregating).** Cada árvore é treinada em uma amostra bootstrap da base, ou seja, *n* linhas sorteadas com reposição. Cerca de 63,2% das linhas originais entram em cada amostra. Os ~36,8% restantes são as amostras **out-of-bag (OOB)**, que servem como validação "de graça".
2. **Subespaço aleatório de atributos.** Em cada nó, a árvore não procura o melhor split entre todos os *p* atributos. Ela sorteia um subconjunto de tamanho `mtry` (no scikit-learn, `max_features`) e escolhe o melhor split só entre eles.

A previsão final é o **voto majoritário** das árvores na classificação (no scikit-learn, a média das probabilidades) e a **média** na regressão.

**Por que funciona.** Uma árvore profunda sozinha tem viés baixo e variância alta, ou seja, sofre overfitting. Breiman mostrou que o erro de generalização da floresta depende de dois fatores: a **força** de cada árvore e a **correlação** entre elas. Ao fazer a média de muitas árvores pouco correlacionadas, a variância cai e o viés quase não aumenta. O `mtry` é justamente o parâmetro que reduz essa correlação. Por isso as árvores de um RF costumam crescer sem poda.

Pontos úteis para justificar a escolha do método:

- Aumentar o número de árvores **não causa overfitting**. O erro converge (teorema em Breiman, 2001).
- A floresta fornece estimativa OOB do erro e medidas de importância de atributos.
- O RF é um dos classificadores mais fortes em dados tabulares. Fernández-Delgado et al. (2014) compararam 179 classificadores em 121 bases da UCI, e o RF ficou entre os melhores. Grinsztajn et al. (2022) mostram que modelos baseados em árvores ainda superam deep learning em dados tabulares de tamanho médio.

## 2. Hiperparâmetros principais

A principal referência é **Probst, Wright & Boulesteix (2019)**, *"Hyperparameters and tuning strategies for random forest"*. Os autores medem a "tunabilidade" de cada hiperparâmetro em dezenas de bases. Os resultados indicam que o RF já funciona razoavelmente bem com os valores padrão, mas que o ajuste traz ganhos, principalmente via `mtry`.

| Hiperparâmetro (sklearn) | O que controla | Recomendação |
|---|---|---|
| **`max_features`** (mtry) | Nº de atributos sorteados por split; controla o trade-off entre correlação e força das árvores | **O mais influente.** O padrão é √p na classificação. Testar valores como `sqrt`, `log2`, 0.2–0.8 |
| **`min_samples_leaf`** / `max_depth` / `min_samples_split` | Tamanho e profundidade das árvores (regularização) | Segundo grupo mais importante. Testar `min_samples_leaf` entre 1 e 20. Em bases ruidosas, folhas maiores ajudam |
| **`max_samples`** + `bootstrap` | Tamanho da amostra de cada árvore | Amostras menores reduzem a correlação e o tempo. Testar 0.5–1.0 |
| **`n_estimators`** | Nº de árvores | **Não é preciso otimizar como os outros.** Usar o maior valor que o tempo permitir; o ganho satura. Oshiro, Perez & Baranauskas (2012) e Probst & Boulesteix (2018) indicam algo entre 64 e 128 árvores como ponto de retornos decrescentes, mas 300–500 é seguro |
| **`class_weight`** | Peso das classes | `balanced` ou `balanced_subsample` em bases desbalanceadas |
| `criterion` | `gini`, `entropy` ou `log_loss` | Impacto geralmente pequeno |

**Estratégia de busca:**

- Usar `RandomizedSearchCV` ou **Optuna** (otimização bayesiana) em vez de grid search exaustivo. Bergstra & Bengio (2012) mostram que a busca aleatória é mais eficiente.
- Usar `StratifiedKFold` e escolher uma métrica coerente com o problema. Em base desbalanceada, preferir F1, recall, ROC-AUC ou PR-AUC em vez de acurácia.
- Para relatar o desempenho de forma honesta, usar **validação cruzada aninhada**: o laço interno faz o ajuste e o laço externo faz a avaliação. Avaliar no mesmo CV usado para o ajuste dá uma estimativa otimista (Cawley & Talbot, 2010).
- O `oob_score=True` serve como estimativa rápida durante a exploração.

## 3. Tratamento da base de dados

O RF dispensa muito do pré-processamento que outros algoritmos exigem.

**O que NÃO é necessário:**

- **Normalização ou padronização.** Os splits dependem só da ordem dos valores, então escala não altera nada.
- **Tratar outliers ou transformar distribuições** (log, Box-Cox). O modelo é robusto a esses casos.
- **Remover atributos correlacionados** para fins de desempenho. Mas ver a observação sobre interpretação no item 5 abaixo.

**O que é importante:**

1. **Valores faltantes.** O scikit-learn aceita `NaN` no RandomForest desde a versão 1.4. Mesmo assim, é comum usar imputação com `SimpleImputer` (mediana) ou `IterativeImputer`/KNN. Atenção a faltantes "disfarçados": na Pima, zeros em glicose, pressão, espessura da pele, IMC e insulina significam ausência de medida.
2. **Atributos categóricos.** O scikit-learn exige números. Usar `OrdinalEncoder` para categorias de baixa cardinalidade ou ordinais e `OneHotEncoder` para nominais. Evitar one-hot com cardinalidade muito alta, porque dilui o sorteio de `mtry`. (Não se aplica à Pima, que só tem atributos numéricos.)
3. **Desbalanceamento de classes.** Há várias opções:
   - `class_weight='balanced_subsample'`;
   - **Balanced Random Forest** (Chen, Liaw & Breiman, 2004), implementado em `imblearn.ensemble.BalancedRandomForestClassifier`, que faz undersampling em cada bootstrap;
   - **SMOTE** (Chawla et al., 2002), aplicado **só nos folds de treino** via `imblearn.pipeline.Pipeline`;
   - ajuste do **limiar de decisão** sobre as probabilidades, em vez do 0,5 padrão.
4. **Evitar vazamento de dados.** Imputação, encoding e reamostragem precisam ficar **dentro do Pipeline**, para serem ajustados apenas no treino de cada fold. Esse é o erro metodológico mais comum.
5. **Seleção de atributos e interpretação.** A importância padrão (MDI, por impureza) é **enviesada** a favor de atributos contínuos ou com muitas categorias (Strobl et al., 2007). Preferir **permutation importance**, calculada no conjunto de teste, ou SHAP. Com atributos muito correlacionados, a importância se divide entre eles, e o relatório deve comentar isso.

**Sugestão experimental:** comparar o RF com uma **árvore de decisão única** na mesma base e com o mesmo protocolo de validação. Isso mostra empiricamente a redução de variância que o bagging promete.

## Referências

- Bergstra, J.; Bengio, Y. (2012). Random search for hyper-parameter optimization. *JMLR*, 13, 281–305.
- **Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.**
- Cawley, G. C.; Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *JMLR*, 11, 2079–2107.
- Chawla, N. V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357.
- Chen, C.; Liaw, A.; Breiman, L. (2004). *Using Random Forest to Learn Imbalanced Data*. Tech. Report 666, UC Berkeley.
- Fernández-Delgado, M. et al. (2014). Do we need hundreds of classifiers to solve real world classification problems? *JMLR*, 15, 3133–3181.
- Grinsztajn, L.; Oyallon, E.; Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on tabular data? *NeurIPS Datasets and Benchmarks*.
- Louppe, G. (2014). *Understanding Random Forests: From Theory to Practice*. Tese de doutorado, Univ. de Liège.
- Oshiro, T. M.; Perez, P. S.; Baranauskas, J. A. (2012). How many trees in a random forest? *MLDM 2012*, LNCS 7376, 154–168.
- **Probst, P.; Wright, M. N.; Boulesteix, A.-L. (2019). Hyperparameters and tuning strategies for random forest. *WIREs Data Mining and Knowledge Discovery*, 9(3), e1301.**
- Probst, P.; Boulesteix, A.-L. (2018). To tune or not to tune the number of trees in random forest. *JMLR*, 18(181), 1–18.
- Strobl, C. et al. (2007). Bias in random forest variable importance measures. *BMC Bioinformatics*, 8, 25.

> As referências em negrito são as essenciais: Breiman (2001) para o método e Probst et al. (2019) para os hiperparâmetros. Conferir volume e páginas no Google Scholar antes de citar.
