import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from model import build_model
from utils import prepare_data, plot_history, compute_accuracy, save_results
import os

# ---------------- LOAD CONFIG ----------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ---------------- MOCK DATA (REPLACE WITH REAL DATASET) ----------------
texts = [
    "She woke up early", 
    "She brushed her teeth", 
    "She went to work", 
    "She had lunch", 
    "She went to sleep"
] * 200

labels = [1,2,3,4,5] * 200

# ---------------- PREPROCESS ----------------
X, y, tokenizer = prepare_data(
    texts,
    labels,
    config['dataset']['vocab_size'],
    config['dataset']['max_len']
)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=config['training']['validation_split']
)

# ---------------- EXPERIMENTS ----------------
experiments = [
    {"name": "baseline", "lstm_units":128, "dropout":0.3},
    {"name": "more_units", "lstm_units":256, "dropout":0.3},
    {"name": "less_units", "lstm_units":64, "dropout":0.3},
    {"name": "high_dropout", "lstm_units":128, "dropout":0.5},
    {"name": "no_dropout", "lstm_units":128, "dropout":0.0},
]

results = []

for exp in experiments:
    print(f"\nRunning {exp['name']}")

    model = build_model(
        vocab_size=config['dataset']['vocab_size'],
        embedding_dim=config['dataset']['embedding_dim'],
        max_len=config['dataset']['max_len'],
        lstm_units=exp['lstm_units'],
        dropout=exp['dropout'],
        num_classes=5
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        verbose=1
    )

    # Save loss curve
    os.makedirs("results/figures", exist_ok=True)
    plot_history(history, f"results/figures/{exp['name']}_loss.png")

    train_acc = compute_accuracy(model, X_train, y_train)
    val_acc = compute_accuracy(model, X_val, y_val)

    result = {
        "experiment": exp['name'],
        "train_loss": history.history['loss'][-1],
        "val_loss": history.history['val_loss'][-1],
        "train_acc": train_acc,
        "val_acc": val_acc
    }

    results.append(result)

# ---------------- SAVE TABLE ----------------
os.makedirs("results/tables", exist_ok=True)
save_results(results, "results/tables/results.txt")

print("\nFinal Results:")
for r in results:
    print(r)