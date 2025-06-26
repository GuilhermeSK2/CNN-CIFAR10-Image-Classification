# Classificação de Imagens com Redes Neurais Convolucionais (CNN) - Dataset CIFAR-10

Este projeto implementa uma **Rede Neural Convolucional (CNN)** para a tarefa de classificação de imagens no famoso dataset **CIFAR-10**. O objetivo é construir e treinar um modelo capaz de reconhecer e classificar corretamente 10 categorias diferentes de objetos em imagens.

---

## O que é o Dataset CIFAR-10?

O **CIFAR-10** é um dataset amplamente utilizado para pesquisa em reconhecimento de objetos. Ele consiste em 60.000 imagens coloridas de 32x32 pixels em 10 classes, com 6.000 imagens por classe. As 10 classes são:

* airplane (avião)
* automobile (automóvel)
* bird (pássaro)
* cat (gato)
* deer (cervo)
* dog (cachorro)
* frog (sapo)
* horse (cavalo)
* ship (navio)
* truck (caminhão)

O dataset é dividido em 50.000 imagens para treinamento e 10.000 imagens para teste.

---

## Estrutura do Projeto (Notebook `CNN.ipynb`)

O notebook contém as seguintes etapas:

1.  **Importação de Bibliotecas:** Importa as bibliotecas necessárias do Keras (para construir a CNN), Matplotlib (para visualização), NumPy (para manipulação de arrays) e Scikit-learn (para métricas de avaliação).
2.  **Carregamento e Exploração dos Dados:**
    * O dataset CIFAR-10 é carregado diretamente do Keras, sendo dividido em conjuntos de treinamento (`x_train`, `y_train`) e teste (`x_test`, `y_test`).
    * Uma imagem de exemplo do conjunto de treinamento é visualizada para inspeção.
    * As dimensões dos dados (`x_train.shape`) e uma amostra dos rótulos (`y_train`) são exibidas para entender a estrutura dos dados.
3.  **Pré-processamento dos Dados:**
    * **Normalização:** Os valores dos pixels das imagens (que variam de 0 a 255) são convertidos para o tipo `float32` e normalizados para o intervalo de 0 a 1, dividindo por 255.0. Isso ajuda no treinamento da rede.
    * **One-Hot Encoding:** Os rótulos das classes (`y_train`, `y_test`) são convertidos para um formato "one-hot encoded". Isso significa que cada rótulo categórico é transformado em um vetor binário, onde apenas a posição correspondente à classe é 1 e as outras são 0. Por exemplo, a classe `6` se torna `[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]`.
4.  **Definição da Arquitetura da CNN:**
    * Um modelo `Sequential` é criado para construir a rede camada por camada.
    * **Camadas Convolucionais (`Conv2D`):** Duas camadas convolucionais com 32 filtros e kernel de 3x3 são adicionadas, seguidas por mais duas com 64 filtros. A função de ativação `relu` (Rectified Linear Unit) é usada. `padding='same'` garante que a saída tenha a mesma dimensão da entrada, enquanto as segundas camadas `Conv2D` não usam `padding='same'`, resultando em uma redução de dimensão.
    * **Camadas de Pooling (`MaxPooling2D`):** Após cada bloco de camadas convolucionais, uma camada `MaxPooling2D` com `pool_size=(2,2)` é usada para reduzir a dimensionalidade espacial (largura e altura) da entrada, ajudando a extrair as características mais importantes e a reduzir a complexidade computacional.
    * **Camadas de Dropout (`Dropout`):** Camadas de `Dropout` com taxa de 0.25 são inseridas após as camadas de pooling e antes da camada de saída densa. O dropout é uma técnica de regularização que desliga aleatoriamente uma porcentagem dos neurônios durante o treinamento para evitar overfitting.
    * **Camada Flatten (`Flatten`):** A saída das camadas convolucionais e de pooling (que é 2D ou 3D) é achatada em um vetor 1D para ser alimentada nas camadas densas.
    * **Camadas Densas (`Dense`):** Duas camadas densas são adicionadas. A primeira tem 512 neurônios e `relu` como função de ativação. A camada final de saída tem 10 neurônios (correspondendo às 10 classes do CIFAR-10) e usa a função de ativação `softmax`, que produz probabilidades para cada classe.
5.  **Compilação e Treinamento do Modelo:**
    * O modelo é compilado com a função de perda (`loss`) `categorical_crossentropy`, que é apropriada para classificação multi-classe one-hot encoded.
    * O otimizador escolhido é `adam`, uma escolha popular e eficiente.
    * A métrica de avaliação definida é `accuracy` (acurácia).
    * O modelo é treinado usando os dados de treinamento por 10 `epochs` (iterações completas sobre todo o conjunto de treinamento) com um `batch_size` de 32 (número de amostras processadas antes de atualizar os pesos do modelo).
6.  **Avaliação do Modelo:**
    * **Previsões:** O modelo treinado faz previsões sobre o conjunto de dados de teste (`x_test`).
    * **Conversão de Previsões e Rótulos:** As previsões (que são probabilidades) e os rótulos de teste (one-hot encoded) são convertidos de volta para o formato de classes inteiras usando `np.argmax`.
    * **Acurácia:** A acurácia do modelo no conjunto de teste é calculada usando `accuracy_score` da Scikit-learn.
    * **Matriz de Confusão:** Uma matriz de confusão é gerada usando `confusion_matrix` para visualizar o desempenho do modelo em cada classe, mostrando os acertos e erros específicos.

---

## Resultados

Após 10 epochs de treinamento (Foram realizadas poucas epochs por fins didáticos para evitar uma demora maior de finalização do treinamento), o modelo atingiu uma **acurácia de aproximadamente 78%** no conjunto de teste, demonstrando boa capacidade de classificação das imagens do CIFAR-10.

A matriz de confusão fornece uma visão detalhada de quais classes estão sendo confundidas pelo modelo.

---

## Como Rodar o Código

Você pode executar este notebook diretamente no **Google Colab** (clicando no ícone "Open in Colab" no GitHub) ou em qualquer ambiente Python que tenha as bibliotecas necessárias instaladas.

### Pré-requisitos:

Certifique-se de ter as seguintes bibliotecas Python instaladas:

```bash
pip install keras tensorflow matplotlib numpy scikit-learn
