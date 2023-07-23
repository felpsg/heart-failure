# Análise de Dados e Modelagem de Previsão de Insuficiência Cardíaca

Este projeto envolve o uso de um dataset de insuficiência cardíaca para realizar análises de dados e construir um modelo de aprendizado de máquina para prever a ocorrência de eventos de morte.

## Análise Exploratória de Dados

As primeiras etapas envolvem a leitura do dataset, a transformação de variáveis categóricas usando LabelEncoder e a conversão de variáveis numéricas para inteiros.

Depois disso, realizamos uma análise exploratória dos dados. Primeiro, criamos gráficos de barras para as variáveis categóricas, analisando a taxa de eventos de morte em relação a cada uma delas. A imagem abaixo mostra um exemplo desse tipo de gráfico. É importante notar como diferentes categorias podem ter diferentes taxas de eventos de morte, o que pode ser uma indicação da importância dessas variáveis para o modelo de previsão.

![Análise de variavéis categoricas x Taxa de eventos de morte](img/figure_analise.png)


Em seguida, criamos uma matriz de correlação para visualizar as relações entre as variáveis numéricas. A matriz de correlação, mostrada abaixo, ajuda a identificar possíveis relações entre diferentes variáveis numéricas. Variáveis que apresentam alta correlação podem influenciar o modelo de maneira semelhante, enquanto aquelas com baixa correlação podem oferecer insights únicos.

![Matriz de correlação](img/Figure_1.png)

## Modelagem de Previsão

Nós então separamos os dados em conjuntos de treinamento e teste, e aplicamos a normalização StandardScaler nas variáveis numéricas. Os rótulos foram preparados para a classificação usando a função `to_categorical`.

Para a modelagem, construímos uma rede neural usando a biblioteca Keras. O modelo tem uma camada de entrada, uma camada oculta com a função de ativação 'relu', e uma camada de saída com a função de ativação 'softmax'. O modelo foi compilado com a perda de entropia cruzada categórica, o otimizador 'adam', e a métrica 'accuracy'.

Depois de treinar o modelo por 100 épocas, a acurácia foi avaliada nos dados de teste. Também geramos um relatório de classificação para ter uma visão mais detalhada do desempenho do modelo. 

Finalmente, criamos gráficos para visualizar a acurácia e a perda durante o treinamento. O gráfico abaixo mostra a evolução da acurácia e da perda ao longo das épocas de treinamento. Tais gráficos são úteis para avaliar o desempenho do modelo ao longo do tempo e identificar possíveis problemas, como overfitting (quando o modelo se ajusta demais aos dados de treinamento) ou underfitting (quando o modelo não se ajusta suficientemente aos dados de treinamento).

![Acurácia do treinamento vs acurácia da validação - Perda do treinamento vs perda da validação](img/acuracia-treinamento.png)
