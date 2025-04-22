import tensorflow as tf
from models.model import Model

class RNN(Model):
    def __init__(self):
        self.model = None

    def build(self, input_shape):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.SimpleRNN(64, activation='relu', return_sequences=False),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def train(self, X, y, epochs=10, batch_size=32):
        return self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)
