#!/usr/bin/env python3

"""Model to classify job descriptions

This file contains all the model information: the training steps, the batch
size and the model itself.
"""

import tensorflow as tf

from trainer import optimization
from trainer.bert import bert
from trainer.nn import neural_network


def model(input_layer, params):
    """Returns a compiled model.

    This function is expected to return a model to identify the different job 
    description classes. The model's outputs are expected to be probabilities 
    for the classes and and it should be ready for training.
    The input layer specifies the shape of the text. 

    Add your model below.

    Parameters:
        input_layer: A tf.keras.layers.InputLayer() specifying the shape of the input.
            tf.string, shape: ()
    Returns:
        model: A compiled model
    """

    if params.model == "nn":
        model = neural_network()
    else:
        model = bert(input_layer, params.model)

    
    num_train_steps = params.steps * params.epochs
    num_warmup_steps = int(0.1*num_train_steps)

    optimizer = optimization.create_optimizer(init_lr=params.init_lr,
                                            num_train_steps=num_train_steps,
                                            num_warmup_steps=num_warmup_steps,
                                            optimizer_type='adam')


    # TODO: Return the compiled model
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        metrics=["accuracy"])

    return model
