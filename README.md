# Análise de Dados e Modelagem de Previsão de Insuficiência Cardíaca

Este projeto envolve o uso de um dataset de insuficiência cardíaca para realizar análises de dados e construir um modelo de aprendizado de máquina para prever a ocorrência de eventos de morte.

## Análise Exploratória de Dados

As primeiras etapas envolvem a leitura do dataset, a transformação de variáveis categóricas usando LabelEncoder e a conversão de variáveis numéricas para inteiros.

Depois disso, realizamos uma análise exploratória dos dados. Primeiro, criamos gráficos de barras para as variáveis categóricas, analisando a taxa de eventos de morte em relação a cada uma delas.

![Análise de variavéis categoricas x Taxa de eventos de morte](img/figure_analise.png)


Em seguida, criamos uma matriz de correlação para visualizar as relações entre as variáveis numéricas.

![Matriz de correlação](img/figure_1.png)

## Modelagem de Previsão

Nós então separamos os dados em conjuntos de treinamento e teste, e aplicamos a normalização StandardScaler nas variáveis numéricas. Os rótulos foram preparados para a classificação usando a função `to_categorical`.

Para a modelagem, construímos uma rede neural usando a biblioteca Keras. O modelo tem uma camada de entrada, uma camada oculta com a função de ativação 'relu', e uma camada de saída com a função de ativação 'softmax'. O modelo foi compilado com a perda de entropia cruzada categórica, o otimizador 'adam', e a métrica 'accuracy'.

Depois de treinar o modelo por 100 épocas, a acurácia foi avaliada nos dados de teste. Também geramos um relatório de classificação para ter uma visão mais detalhada do desempenho do modelo.

Finalmente, criamos gráficos para visualizar a acurácia e a perda durante o treinamento.

![Acurácia do treinamento vs acurácia da validação - Perda do treinamento vs perda da validação](img/acuracia-treinamento.png)

