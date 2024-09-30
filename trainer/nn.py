#!/usr/bin/env python3

"""Neural network to classify job descriptions

This file contains the definition of the neural network itself
as well as a pre-trained embedding from TensorFlow hub.
"""

import tensorflow as tf
import tensorflow_hub as hub

def neural_network():
    embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
    hub_layer = hub.KerasLayer(embedding, dtype=tf.string, trainable=True)
    model = tf.keras.Sequential([
        hub_layer,
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(5)
    ])

    return model