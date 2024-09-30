#!/usr/bin/env python3

"""Train and evaluate the model

This file trains the model upon the training data and evaluates it with
the eval data.
"""

import os
import yaml
from datetime import datetime
import argparse
import logging
import math
import tensorflow as tf
import keras

import trainer.data as data
import trainer.model as model

from utils.plot import plot_metrics


def get_args_parser():
    parser = argparse.ArgumentParser('Job Classification', add_help=False)

    # Training parameters
    parser.add_argument('--model', default='small_bert/bert_en_uncased_L-4_H-512_A-8')
    parser.add_argument('--init-lr', default=3e-5, type=float, help="Initial learning rate")
    parser.add_argument('--batch-size', default=32, type=int, help="Batch size")
    parser.add_argument('--epochs', default=5, type=int, help="Number of epochs")
    parser.add_argument('--shuffle-buffer-size', default=100, type=int, help="Shuffle buffer size")

    return parser

def train_model(params):
    """The function gets the training data from the training folder,
    the evaluation data from the eval folder and trains the model
    from the model.py file with it.

    Parameters:
        params: parameters for training the model
    """

    # create results directory
    results_dir = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(results_dir)

    AUTOTUNE = tf.data.AUTOTUNE

    # load data
    (train_data, train_labels) = data.load_dataset("data/train.csv")
    (eval_data, eval_labels) = data.load_dataset("data/eval.csv")

    # preprocess data
    train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    eval_ds = tf.data.Dataset.from_tensor_slices((eval_data, eval_labels))
    train_ds = train_ds.shuffle(params.shuffle_buffer_size).batch(params.batch_size)
    eval_ds = eval_ds.shuffle(params.shuffle_buffer_size).batch(params.batch_size)
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    eval_ds = eval_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # initialize optimizer parameters
    params.steps = math.ceil(len(train_data)/params.batch_size)

    # initialize model
    input_layer = tf.keras.Input(shape=(), name='input_text', dtype=tf.string)
    ml_model = model.model(input_layer, params)

    # initialize tensorboard
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=results_dir)

    # train model
    if ml_model is None:
        print("No model found. You need to implement one in model.py")
    else:
        hist = ml_model.fit(x=train_ds,
                     epochs=params.epochs,
                     validation_data=eval_ds,
                     callbacks=[tensorboard_callback])
        _ = ml_model.evaluate(eval_ds, verbose=1)
    
    # plot metrics
    plot_metrics(hist.history, savepath=results_dir)
        
    # save config as yaml
    with open(os.path.join(results_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(params), f, default_flow_style=False)

    # save model
    ml_model.save(os.path.join(results_dir, "model.keras"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Job Classification", parents=[get_args_parser()])

    args = parser.parse_args()
    tf_logger = logging.getLogger("tensorflow")
    tf_logger.setLevel(logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf_logger.level // 10)

    train_model(args)
