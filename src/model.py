import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(vocab_size, embedding_dim, max_len, lstm_units, dropout, num_classes):
    model = models.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
        layers.LSTM(lstm_units),
        layers.Dropout(dropout),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model