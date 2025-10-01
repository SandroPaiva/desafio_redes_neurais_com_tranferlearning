# desafio_redes_neurais_com_tranferlearning
Aplicação do método de Transfer Learning em uma rede de Deep Learning na linguagem Python no ambiente COLAB. 

# Estudo de Aprendizado por Transferência com VGG16 e Keras

Este repositório contém um estudo prático sobre **Aprendizado por Transferência** (*Transfer Learning*) e **Ajuste Fino** (*Fine-tuning*) utilizando a biblioteca Keras com backend TensorFlow. O projeto foi desenvolvido em um ambiente Google Colab e baseia-se em um tutorial clássico de classificação de imagens.

A principal alteração realizada neste projeto foi a substituição do dataset original `101_ObjectCategories` por um conjunto de imagens customizado da série **Kamen Rider**, com as categorias **Kabuto** e **Zeztz**. Todo o restante da estrutura do projeto foi mantido como no original, com a intenção plenamente acadêmica de explorar e compreender a técnica de Transfer Learning.

## 🧠 Conceitos Principais

### Aprendizado por Transferência / Ajuste Fino

De modo geral, o aprendizado por transferência refere-se ao processo de aproveitar o conhecimento aprendido em um modelo para o treinamento de outro. O processo envolve pegar uma rede neural existente, previamente treinada em um conjunto de dados maior (como o ImageNet), e usá-la como base para um novo modelo. Esse método é extremamente popular e eficaz para melhorar o desempenho de uma rede treinada em um conjunto de dados pequeno. A intuição é que as camadas iniciais de uma rede convolucional aprendem características genéricas (como bordas, texturas e cores) que são úteis para diversas tarefas de visão computacional.

### Extração de Características vs. Ajuste Fino

Existem duas estratégias principais no aprendizado por transferência:

1.  **Extração de Características:** A rede pré-treinada é usada como um extrator de características fixo. Os pesos de suas camadas são "congelados" (não são atualizados durante o treinamento), e apenas os pesos da nova camada de classificação, adicionada no topo, são treinados com o novo dataset.
2.  **Ajuste Fino (Fine-tuning):** Além de treinar a nova camada de classificação, permitimos que os pesos de algumas das camadas superiores da rede pré-treinada também sejam atualizados, geralmente com uma taxa de aprendizado baixa. Isso "ajusta" as características mais especializadas da rede para a nova tarefa.

## 🚀 Procedimento Adotado

Neste estudo, utilizamos a abordagem de **extração de características**. Carregamos o modelo **VGG16**, treinado no dataset ImageNet, e seguimos os seguintes passos:

1.  **Carregamento dos Dados:** O dataset original `101_ObjectCategories` foi substituído por um conjunto de imagens com as categorias:
      * `KamenRider/Kabuto`
      * `KamenRider/Zeztz`
2.  **Pré-processamento:** As imagens foram redimensionadas para `224x224` pixels, o formato de entrada esperado pelo VGG16.
3.  **Divisão do Dataset:** Os dados foram divididos em conjuntos de treino (70%), validação (15%) e teste (15%).

### 1\. Linha de Base: Treinando uma CNN do Zero

Para termos uma base de comparação, primeiro foi criada e treinada uma Rede Neural Convolucional (CNN) simples, do zero. A arquitetura foi a seguinte:

```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 222, 222, 32)   │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ ... (demais camadas)            │ ...                    │           ... │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 3)              │           771 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation_5 (Activation)       │ (None, 3)              │             0 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 1,209,315 (4.61 MB)
 Trainable params: 1,209,315 (4.61 MB)
 Non-trainable params: 0 (0.00 B)
```

  - **Resultado:** A precisão final no conjunto de teste foi de aproximadamente **50%**.

### 2\. Abordagem Principal: Transfer Learning com VGG16

Em seguida, a estratégia de aprendizado por transferência foi aplicada:

1.  O modelo **VGG16** foi carregado com seus pesos pré-treinados no ImageNet.
2.  A camada de classificação final (`predictions`), que originalmente classificava 1000 classes, foi removida.
3.  Uma nova camada `Dense` com ativação `softmax` foi adicionada para classificar nossas novas categorias.
4.  Todas as camadas do VGG16 foram **congeladas** (`layer.trainable = False`), de modo que apenas a nova camada de classificação fosse treinada.

<!-- end list -->

```
Model: "functional_20"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_1 (InputLayer)      │ (None, 224, 224, 3)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ ... (camadas do VGG16)          │ ...                    │           ... │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ fc2 (Dense)                     │ (None, 4096)           │    16,781,312 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 3)              │        12,291 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 134,272,835 (512.21 MB)
 Trainable params: 12,291 (48.01 KB)
 Non-trainable params: 134,260,544 (512.16 MB)
```

  - **Resultado:** A precisão final no conjunto de teste foi de **50%**, embora a acurácia de validação durante o treino tenha alcançado picos significativamente mais altos, demonstrando a capacidade do modelo de aprender características relevantes muito mais rapidamente.

## 📊 Resultados e Comparação

O gráfico abaixo compara a acurácia de validação ao longo das épocas para os dois modelos (azul: CNN do zero; laranja: VGG16 com Transfer Learning).

<img width="1291" height="393" alt="image" src="https://github.com/user-attachments/assets/731bab90-5a58-494e-b566-dec6254d1f0c" />


Fica evidente que a abordagem de Transfer Learning (curva verde) atinge uma acurácia de validação superior e mais estável em comparação com o modelo treinado do zero, que demonstra sinais de sobreajuste (*overfitting*) e dificuldade em aprender com o pequeno conjunto de dados.

## 🛠️ Como Executar

O código foi desenvolvido em um notebook Google Colab. As principais dependências são:

  * `tensorflow` / `keras`
  * `numpy`
  * `matplotlib`
  * `Pillow`

Basta abrir o arquivo `.ipynb` no Google Colab ou em um ambiente Jupyter e executar as células em sequência. O dataset `KamenRider` deve estar na mesma estrutura de pastas mencionada no código.
## Possíveis Melhorias

* **Aumento de Dados (Data Augmentation)**: Aplicar transformações aleatórias (rotações, zooms, flips) nas imagens de treino para aumentar a variabilidade do dataset e reduzir o overfitting.
* **Ajuste Fino de Mais Camadas**: Descongelar algumas das últimas camadas do VGG16 e treiná-las com uma taxa de aprendizado muito baixa (`low learning rate`).
* **Otimizadores e Hiperparâmetros**: Experimentar diferentes otimizadores (ex: SGD, RMSprop) e ajustar hiperparâmetros como a taxa de aprendizado e o tamanho do lote (`batch size`). 
