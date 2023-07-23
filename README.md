# Insuficiência Cardíaca - Análise de Dados e Modelagem Preditiva

Este repositório foca-se na Análise Exploratória de Dados (EDA) e na construção de um modelo preditivo para eventos de morte relacionados à insuficiência cardíaca, utilizando um conjunto de dados específico.

O projeto envolve as seguintes etapas e ferramentas:

- Manipulação de dados com a biblioteca **Pandas**
- Visualização de dados com as bibliotecas **Seaborn** e **Matplotlib**
- Pré-processamento de dados, incluindo codificação de rótulos e normalização de colunas numéricas
- Divisão de dados em conjuntos de treinamento e teste
- Criação e treinamento de uma rede neural profunda usando a biblioteca **Keras**
- Avaliação do modelo e geração de um relatório de classificação

## Exploração e Processamento de Dados

A análise inicial do dataset envolve a manipulação e preparação dos dados através de técnicas como Label Encoding para variáveis categóricas e a normalização de colunas numéricas.

A EDA é realizada através de gráficos de barras e uma matriz de correlação, que permitem uma visualização mais clara das relações entre as variáveis.

![Taxa de Eventos de Morte por Categorias](img/figure_analise.png)
![Matriz de Correlação](img/Figure_1.png)

## Modelagem Preditiva

Após a divisão dos dados em conjuntos de treinamento e teste, implementamos uma Rede Neural Profunda utilizando a biblioteca Keras. Com uma camada de entrada, uma camada oculta (com função de ativação 'relu'), e uma camada de saída (com função de ativação 'softmax'), o modelo é compilado com Categorical Cross-Entropy Loss, otimizador 'Adam' e métrica 'accuracy'.

O modelo é então treinado por 100 épocas e avaliado no conjunto de teste, com a geração de um relatório de classificação detalhado.

A evolução do treinamento é visualizada através de gráficos de acurácia e perda, úteis para identificar possíveis overfitting ou underfitting.

![Acurácia do Treinamento vs Acurácia da Validação - Perda do Treinamento vs Perda da Validação](img/acuracia-treinamento.png)
