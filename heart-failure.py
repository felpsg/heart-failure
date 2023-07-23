import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.utils import to_categorical

# Carregando os dados
dados = pd.read_csv('./heart_failure.csv')  # Lendo os dados do arquivo CSV

# Aplicando o LabelEncoder em cada uma das colunas categóricas
codificador = LabelEncoder()
dados['anaemia'] = codificador.fit_transform(dados['anaemia'])
dados['diabetes'] = codificador.fit_transform(dados['diabetes'])
dados['high_blood_pressure'] = codificador.fit_transform(dados['high_blood_pressure'])
dados['sex'] = codificador.fit_transform(dados['sex'])
dados['smoking'] = codificador.fit_transform(dados['smoking'])
dados['death_event'] = codificador.fit_transform(dados['death_event'])

# Convertendo colunas numéricas para inteiros
for coluna in dados.columns:
    if dados[coluna].dtype == 'float64':
        dados[coluna] = dados[coluna].astype(int)

# Criando gráficos de barras para as variáveis categóricas
variaveis_categoricas = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
fig.suptitle('Análise de Variáveis Categóricas em relação à Taxa de Eventos de Morte')

for i, variavel in enumerate(variaveis_categoricas):
    sns.barplot(x=variavel, y='death_event', data=dados, ax=axes[i // 3, i % 3])
    axes[i // 3, i % 3].set_title(f'Taxa de eventos de morte em relação a {variavel}')
    axes[i // 3, i % 3].set_xlabel(variavel)
    axes[i // 3, i % 3].set_ylabel('Taxa de eventos de morte')

plt.show()

# Criando uma matriz de correlação para as variáveis numéricas
plt.figure(figsize=(12, 10))
sns.heatmap(dados.corr(), annot=True, fmt=".2f")
plt.title('Matriz de Correlação')
plt.show()

# Separando dados de treinamento e dados de teste
X = dados[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']]  # Atributos
Y = dados['death_event']  # Rótulo

# Pré-processamento de dados
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.3, random_state=0)  # Dividindo os dados em conjuntos de treinamento e teste

# Normalizando as colunas numéricas
ct = ColumnTransformer([("numeric", StandardScaler(), ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time'])])
X_treino = ct.fit_transform(X_treino)  # Ajustando e transformando os dados de treinamento
X_teste = ct.transform(X_teste)  # Transformando os dados de teste

# Preparando rótulos para a classificação
Y_treino = to_categorical(Y_treino)  
Y_teste = to_categorical(Y_teste)

# Construindo o modelo
modelo = Sequential()  # Criando uma rede neural sequencial
modelo.add(InputLayer(input_shape=(X_treino.shape[1],)))  # Adicionando a camada de entrada
modelo.add(Dense(12, activation='relu'))  # Adicionando uma camada oculta com a função de ativação 'relu'
modelo.add(Dense(2, activation='softmax'))  # Adicionando a camada de saída com a função de ativação 'softmax'

# Compilando o modelo
modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # Compilando o modelo com a perda de entropia cruzada categórica, o otimizador 'adam' e a métrica 'accuracy'

# Treinando e avaliando o modelo
historico = modelo.fit(X_treino, Y_treino, epochs=100, batch_size=16, verbose=1, validation_data=(X_teste, Y_teste))  # Treinando o modelo e armazenando os dados de treinamento em 'historico'
perda, acuracia = modelo.evaluate(X_teste, Y_teste, verbose=1)  # Avaliando o modelo

# Gerando um relatório de classificação
Y_estimado = modelo.predict(X_teste)  # Fazendo previsões com os dados de teste
Y_estimado = np.argmax(Y_estimado, axis=1)  # Escolhendo a classe com a maior probabilidade
Y_verdadeiro = np.argmax(Y_teste, axis=1)  # Verdadeiros rótulos

# Criando um gráfico para visualizar a acurácia e a perda durante o treinamento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(historico.history['accuracy'], label='acurácia do treinamento')
plt.plot(historico.history['val_accuracy'], label='acurácia da validação')
plt.title('Acurácia do Treinamento vs Acurácia da Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(historico.history['loss'], label='perda do treinamento')
plt.plot(historico.history['val_loss'], label='perda da validação')
plt.title('Perda do Treinamento vs Perda da Validação')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.show()

