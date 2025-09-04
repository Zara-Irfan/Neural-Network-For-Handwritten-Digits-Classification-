# Neural-Network-For-Handwritten-Digits-Classification
# MNIST Handwritten Digit Recognition

This project demonstrates a simple neural network for recognizing handwritten digits from the MNIST dataset using TensorFlow and Keras. The model is trained to classify images of digits (0-9) and visualizes predictions along with a confusion matrix.

---

## Table of Contents

- [Dataset](#dataset)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Model Architecture](#model-architecture)  
- [Results](#results)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Dataset

The MNIST dataset contains 70,000 grayscale images of handwritten digits, each of size 28x28 pixels:

- Training set: 60,000 images  
- Test set: 10,000 images  

Labels are integers from 0 to 9 representing the digit.

---

## Features

- Normalize and flatten input images for neural network training.  
- Fully connected neural network with one hidden layer using ReLU activation.  
- Softmax output layer for multi-class classification.  
- Train and evaluate model on the MNIST dataset.  
- Display sample predictions with true and predicted labels.  
- Generate a confusion matrix heatmap to visualize model performance.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/mnist-digit-recognition.git
cd mnist-digit-recognition
