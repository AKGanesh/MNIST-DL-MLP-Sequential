![Logo](https://github.com/AKGanesh/MNIST-DL-MLP-Sequential/blob/main/mnist.png)

# Fashion MNIST Clothing Classification - Using DeepLearning

The Fashion-MNIST clothing classification problem is a standard dataset used in computer vision and deep learning.

The dataset is relatively simple, it can be used as the basis for learning and practicing how to develop, evaluate, and use neural networks for image classification from scratch.
https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist

## Implementation Details

- Dataset: Load the dataset from the above link
- Model: Sequential
- Input: Test dataset
- Output: Test loss, Test Accuracy
- Scores: Accuracy and cross validation score
- Loss: sparse_categorical_crossentropy
- Optimizer: Adam
- Others: How to deal with images data, tensorflow, keras, tensorboard, MLP

## Dataset details

This is a dataset of 60,000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images. This dataset can be used as a drop-in replacement for MNIST.

## Process

- Data and Pre-processing

  - Import the required libraries
  - Read the dataset
  - Preprocessing (Normalize 0-1)
  - Data Preparation (flatten)

- Model Development
  - Create a Sequential model (Input, 4 dense hidden, Output)
- Compile and Test
  - Set loss, Optimizer and metrics
  - Check the model summary
  - Train the model (epcohs,batchsize, validation and callbacks)
  - Callbacks are to log to tensorboard
  - Generate the plots - Loss vs Epochs for Train and validation set
  - Evaluate the model on test set and check the results

## Evaluation and Results

| NN  | Params | Test Loss, Test Acc |
| --- | ------ | ------------------- |

|Input(shape=(28, 28, 1)),
|Flatten(),
|Dense(64, activation='relu'),
|Dense(128, activation='relu'),
Dense(64, activation='relu'),
Dense(32, activation='relu'),
Dense(10, activation='softmax') | 260k | [0.4220083951950073, 0.8873999714851379]|
| Input(shape=(28, 28, 1)),
Flatten(),
Dense(64, activation='relu'),
Dense(128, activation='relu'),
Dense(32, activation='relu'),
Dense(10, activation='softmax') |63k | [0.38322576880455017, 0.8853999972343445]|
| Input(shape=(28, 28, 1)),
Flatten(),
Dense(64, activation='relu'),
Dense(32, activation='relu'),
Dense(10, activation='softmax') |52k | [0.3701764941215515, 0.8819000124931335]|
| Input(shape=(28, 28, 1)),
Flatten(),
Dense(128, activation='relu'),
Dense(64, activation='relu'),
Dense(10, activation='softmax') |109k | [0.35463255643844604, 0.8860999941825867]|

## Observations

- Tested the scenarios of changing the number of layers and perceptrons on the layer.
- This is an exercise to choose the right combination of layers and perceptrons based on the loss and accuracy.

## Libraries

**Language:** Python,

**Packages:** Tensorflow, Keras, Numpy, Matplotlib, Tensorboard

## Roadmap

- To check with CNN for image classification in DeepLearning
- To work with checkpoints, other callbacks in tensorboard

## FAQ

#### Whats is Tensorboard?

In machine learning, to improve something you often need to be able to measure it. TensorBoard is a tool for providing the measurements and visualizations needed during the machine learning workflow. It enables tracking experiment metrics like loss and accuracy, visualizing the model graph, projecting embeddings to a lower dimensional space, and much more.
https://www.tensorflow.org/tensorboard/get_started

#### What is a classification problem?

A classification problem in machine learning is a task where the goal is to predict the category (or class) that a new data point belongs to. It's essentially about sorting things into predefined groups. Different types include Binary, Multi-Class and Multi-Label.

#### What is Keras?

Keras is a deep learning API written in Python and capable of running on top of either JAX, TensorFlow, or PyTorch. As a multi-framework API, Keras can be used to develop modular components that are compatible with any framework – JAX, TensorFlow, or PyTorch.

## Acknowledgements

- https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data
- https://www.tensorflow.org/tensorboard
- https://keras.io/guides/sequential_model/

## Contact

For any queries, please send an email (id on github profile)

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
