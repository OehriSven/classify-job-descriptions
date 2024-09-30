"""Plot training metrics

This file contains the method that visualizes the training metrics.
"""
import os
import matplotlib.pyplot as plt

def plot_metrics(history, savepath):
    # create epoch list
    epochs = range(1, len(history["loss"]) + 1)

    # derive max accuracy
    idx_max_acc = max(range(len(history["val_accuracy"])), key=history["val_accuracy"].__getitem__)

    # plot loss
    plt.figure(1)
    plt.plot(epochs, history["loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Validation")
    plt.grid(which="both")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.savefig(os.path.join(savepath, "loss.png"), bbox_inches="tight")

    # plot accuracy
    plt.figure(2)
    plt.plot(epochs, history["accuracy"], label="Train")
    plt.plot(epochs, history["val_accuracy"], label="Validation")
    plt.grid(which="both")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy [%]")
    plt.title("Accuracy")
    plt.legend()
    plt.figtext(0.125,
                -0.02,
                f"Max. Validation Accuracy of {history['val_accuracy'][idx_max_acc]*100:.2f}% (@ Epoch {idx_max_acc + 1})", ha="left")
    plt.savefig(os.path.join(savepath, "accuracy.png"), bbox_inches="tight")