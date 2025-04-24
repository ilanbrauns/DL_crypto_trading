import tensorflow as tf
from models.model import Model

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, attention_axes=1)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attention_output = self.attention(inputs, inputs)
        x = self.norm1(inputs + attention_output)
        ffn_output = self.ffn(x)
        return self.norm2(x + ffn_output)

class Transformer(Model):
    def __init__(self):
        self.model = None

    def build(self, input_shape):
        time_steps, features = input_shape
        embed_dim = 32
        num_heads = 2
        ff_dim = 32

        inputs = tf.keras.Input(shape=(time_steps, features), dtype=tf.float32)
        x = tf.keras.layers.Dense(embed_dim)(inputs)
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1)(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def train(self, X, y, epochs=2, batch_size=32):
        return self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)
