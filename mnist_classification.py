# -------------------------------
# MNIST Handwritten Digit Recognition
# -------------------------------

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

# For inline plots in Jupyter
%matplotlib inline

# -------------------------------
# Load the MNIST dataset
# -------------------------------
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# -------------------------------
# Explore data
# -------------------------------
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Shape of one image: {X_train[0].shape}")

plt.matshow(X_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.axis('off')
plt.show()

# -------------------------------
# Preprocess data
# -------------------------------
# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten images for fully connected network
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)

# -------------------------------
# Build the neural network
# -------------------------------
model = keras.Sequential([
    keras.layers.Dense(128, input_shape=(784,), activation='relu'),  # Hidden layer
    keras.layers.Dense(10, activation='softmax')                      # Output layer for 10 classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# Train the model
# -------------------------------
model.fit(X_train_flattened, y_train, epochs=5, batch_size=32)

# -------------------------------
# Evaluate the model
# -------------------------------
test_loss, test_accuracy = model.evaluate(X_test_flattened, y_test)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# -------------------------------
# Make predictions
# -------------------------------
y_predicted = model.predict(X_test_flattened)
y_predicted_labels = np.argmax(y_predicted, axis=1)

# -------------------------------
# Display sample predictions
# -------------------------------
plt.figure(figsize=(10,4))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.matshow(X_test[i], cmap='gray', fignum=1)
    plt.title(f"True: {y_test[i]}\nPred: {y_predicted_labels[i]}")
    plt.axis('off')
plt.show()

# -------------------------------
# Confusion matrix
# -------------------------------
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("MNIST Confusion Matrix")
plt.show()
