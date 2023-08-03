import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Define the model architecture
model = Sequential()

model.add(Dense(64, activation="relu", input_shape=(10,)))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Generate random training data

x_train = np.random.random((1000, 10))
y_train = np.random.randint(2, size=(1000, 1))

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Generate random test data
x_test = np.random.random((100, 10))
y_test = np.random.randint(2, size=(100, 1))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)

print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
