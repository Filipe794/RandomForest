# Random Forest na base Pima Indians Diabetes

Trabalho da disciplina **Aprendizado de Máquina**, do **PPGCC – UFPI**, 2026.2.

**Autores:** Filipe Mendes, Bétanny Alexandra, Juciane Maciel
**Orientação/professor:** Prof. Dr. Vinicius Ponte Machado

---

## Sobre o projeto

Este projeto aplica o algoritmo **Random Forest** (Breiman, 2001) à base **Pima Indians Diabetes** para classificar se uma paciente tem diagnóstico de diabetes. O trabalho dá continuidade ao estudo anterior da disciplina com árvores de decisão. Ao usar a mesma base, é possível comparar uma árvore única com um ensemble de árvores sob o mesmo protocolo experimental.

### Objetivos

- Estudar o funcionamento do Random Forest: bagging, sorteio aleatório de atributos por split e agregação por votação.
- Investigar o efeito dos principais hiperparâmetros (`max_features`, `min_samples_leaf`/`max_depth`, `max_samples`, `n_estimators`, `class_weight`) nas métricas do modelo.
- Definir um tratamento adequado da base, com atenção aos valores faltantes e ao desbalanceamento de classes.
- Avaliar o modelo com validação cruzada estratificada e comparar com a árvore de decisão única.

### A base de dados

A Pima Indians Diabetes vem do National Institute of Diabetes and Digestive and Kidney Diseases e está disponível no UCI/Kaggle. Ela reúne registros de mulheres de ascendência Pima com pelo menos 21 anos.

- **768 instâncias** e **8 atributos numéricos**: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`.
- **Classe (`Outcome`)**: 0 = sem diabetes (500 casos), 1 = com diabetes (268 casos). A base é moderadamente desbalanceada, com cerca de 35% de positivos.
- **Faltantes disfarçados:** em `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` e `BMI`, o valor zero indica ausência de medida, não um valor real.

## Estrutura do repositório

```
RandomForest/
├── README.md                                 # este arquivo
└── docs/
    ├── fundamentacao-random-forest.md        # funcionamento, hiperparâmetros e tratamento da base
    └── arvores-nao-binarias.md               # por que florestas quase sempre usam árvores binárias
```

## Documentação

- [Fundamentação teórica do Random Forest](docs/fundamentacao-random-forest.md): como o algoritmo funciona, quais hiperparâmetros ajustar e quais tratamentos de dados usar, com referências.
- [Árvores não binárias em Random Forest](docs/arvores-nao-binarias.md): bibliotecas que oferecem splits multiway e as razões pelas quais o split binário é o padrão.

## Referência principal

Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
