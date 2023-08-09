import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert labels to one-hot vectors
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Create the neural network model
model = keras.Sequential(
    [
        layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 images to a 1D array
        layers.Dense(
            128, activation="relu"
        ),  # Fully connected layer with 128 units and ReLU activation
        layers.Dropout(0.3),  # Dropout to prevent overfitting
        layers.Dense(
            10, activation="softmax"
        ),  # Output layer with 10 units for 10 classes and softmax activation
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Set up TensorBoard
tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)

# Print the model summary
model.summary()

# Train the model with TensorBoard callback
BATCH_SIZE = 32
EPOCHS = 50
model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    callbacks=[tensorboard_callback],
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy}")
