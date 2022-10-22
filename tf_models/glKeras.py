from unicodedata import name
import tensorflow.compat.v1 as tf

from tensorflow.keras import layers, Model

class ModelGL(Model):  # model.fit, model.evalute, model.predict
    def __init__(self, input_dim=10, hidden_dim=10, output_dim=10):
        super(ModelGL, self).__init__()
        self.dense1 = tf.keras.layers.Dense(input_dim)
        self.dense2 = tf.keras.layers.Dense(hidden_dim)
        self.dense3 = tf.keras.layers.Dense(hidden_dim)
        self.dense4 = tf.keras.layers.Dense(hidden_dim)
        self.last_layer = Last_Layer_GBar(output_dim)
        
    def call(self, input_tensor):
        # sigmoid version
        # x_1 = tf.sigmoid(self.dense1(input_tensor))
        # x_2 = tf.sigmoid(self.dense2(x_1))
        # x_3 = tf.sigmoid(self.dense3(x_2))
        # x_4 = tf.sigmoid(self.dense4(x_3))

        x_1 = tf.keras.activations.swish(self.dense1(input_tensor))
        x_2 = tf.keras.activations.swish(self.dense2(x_1))
        x_3 = tf.keras.activations.swish(self.dense3(x_2))
        x_4 = tf.keras.activations.swish(self.dense4(x_3))

        return self.last_layer(x_4, input_tensor)

class Last_Layer(layers.Layer):

    def __init__(self, units):
        super(Last_Layer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name='w',
            shape=(input_shape[-1],self.units),
            initializer='random_normal',
            trainable=True,
        )

        self.b = self.add_weight(
            name='b',shape=(self.units,),initializer='zeros',trainable=True,
        )

    def call(self, inputs):
        return tf.matmul(inputs,self.w) + self.b


class Last_Layer_GBar(layers.Layer):

    def __init__(self, units):
        super(Last_Layer_GBar, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name='w',
            shape=(input_shape[-1],self.units),
            initializer='random_normal',
            trainable=True,
        )

        self.b = self.add_weight(
            name='b',shape=(self.units,),initializer='zeros',trainable=True,
        )

    def call(self, inputs, inputNN):
        factor = tf.math.multiply(inputNN, tf.constant([0, 0, 0, 0, 1, -1], dtype=tf.float32))
        factor = tf.reduce_sum(factor)
        return tf.math.multiply((tf.matmul(inputs,self.w) + self.b), tf.stack([tf.constant(1.0, dtype=tf.float32), factor, factor, factor, factor]))