# Árvores não binárias em Random Forest

## Bibliotecas com splits multiway

Quase todas as implementações populares (scikit-learn, `randomForest` e `ranger` no R, Spark MLlib, H2O, cuML) usam árvores **CART binárias**, seguindo o desenho original de Breiman (2001). As opções com divisões multiway (não binárias) são poucas:

| Biblioteca | Linguagem | Como faz o split não binário |
|---|---|---|
| **Weka** (`RandomForest` → `RandomTree`) | Java (tem GUI e pode ser chamado do Python via `python-weka-wrapper3`) | Atributos **nominais** geram **um ramo por valor**. Atributos numéricos continuam com split binário por limiar |
| **chefboost** | Python | Oferece random forest (`enableRandomForest`) sobre árvores **ID3, C4.5 ou CHAID**, que fazem split multiway em atributos categóricos. É mais didática e lenta, mas útil para comparação |
| **Montagem própria** | Python/R | Bagging com subespaço aleatório sobre uma árvore multiway, por exemplo um C4.5 compatível com scikit-learn dentro de `BaggingClassifier(max_features=...)` |

Observações:

- No R, `party`/`partykit::cforest` usa árvores de inferência condicional. Elas ainda são binárias, mas os splits têm seleção não enviesada de variáveis. O pacote `CHAID` faz árvores multiway, mas não tem floresta pronta.
- O Orange tem árvore multiway, mas o RandomForest dele usa o scikit-learn por baixo, então é binário.

## Por que árvores não binárias são pouco usadas

1. **Equivalência.** Qualquer split multiway pode ser representado por uma sequência de splits binários. O split binário não perde poder de representação.
2. **Fragmentação dos dados.** Splits multiway dividem a base em muitos pedaços logo nos primeiros níveis, e os nós seguintes ficam com poucas amostras para estimar novos splits. Hastie, Tibshirani & Friedman (2009, seção 9.2.4) discutem esse problema e concluem que splits binários são preferíveis.
3. **Viés de seleção.** Critérios como o ganho de informação favorecem atributos com muitas categorias, porque eles geram mais partições e, por consequência, nós mais "puros" por acaso. Foi por isso que Quinlan (1993) introduziu o *gain ratio* no C4.5. Esse viés se soma ao das importâncias por impureza descrito por Strobl et al. (2007).

## Relevância para este trabalho

A base Pima tem apenas atributos **numéricos**. Neste caso, mesmo as implementações "multiway" (como a do Weka) fazem splits binários por limiar. Na prática, uma floresta binária e uma multiway se comportam de forma praticamente igual nessa base. Uma comparação entre as duas abordagens só faria sentido em uma base com atributos categóricos.

## Referências

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Hastie, T.; Tibshirani, R.; Friedman, J. (2009). *The Elements of Statistical Learning*. 2. ed. Springer.
- Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.
- Strobl, C. et al. (2007). Bias in random forest variable importance measures. *BMC Bioinformatics*, 8, 25.
