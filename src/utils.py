import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import os

def prepare_data(texts, labels, vocab_size, max_len):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')

    labels = np.array(labels) - 1  # convert 1-5 → 0-4

    return padded, labels, tokenizer


def plot_history(history, save_path):
    plt.figure()

    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')

    plt.legend()
    plt.title("Loss Curve")

    plt.savefig(save_path)
    plt.close()


def save_results(results, path):
    with open(path, "w") as f:
        for r in results:
            f.write(str(r) + "\n")


def compute_accuracy(model, X, y):
    preds = model.predict(X)
    preds = np.argmax(preds, axis=1)
    acc = np.mean(preds == y)
    return acc