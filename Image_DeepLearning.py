import tensorflow as tf
from tensorflow.keras import layers, models

# Load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalise data
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

# Build model
model = models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation="softmax"))

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(X_train, y_train, epochs=5)

# Evaluate
model.evaluate(X_test, y_test)

