import os
import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.config import (
    BEST_MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    TRAINING_CURVES_PATH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE
)
from src.dataset import prepare_dataset_splits
from src.models import get_model

# ==============================================================================
# Plot Training Curves
# ==============================================================================

def plot_training_history(history, save_path=TRAINING_CURVES_PATH):
    """Save clean accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history.history["accuracy"]) + 1)
    
    # Accuracy Plot
    ax1.plot(epochs, history.history["accuracy"], label="Training Accuracy", color="#2563eb", linewidth=2)
    ax1.plot(epochs, history.history["val_accuracy"], label="Validation Accuracy", color="#10b981", linewidth=2, linestyle="--")
    ax1.set_title("Model Accuracy Across Epochs", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Loss Plot
    ax2.plot(epochs, history.history["loss"], label="Training Loss", color="#dc2626", linewidth=2)
    ax2.plot(epochs, history.history["val_loss"], label="Validation Loss", color="#f59e0b", linewidth=2, linestyle="--")
    ax2.set_title("Model Loss Across Epochs", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training history curves saved to {save_path}")

# ==============================================================================
# Training Pipeline
# ==============================================================================

def train_pipeline(
    model_type: str = "cnn",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    split_strategy: str = "stratified"
):
    """Run end-to-end dataset preparation, model compilation, training, and evaluation."""
    feature_type = "1d" if model_type.lower() in ["cnn", "1d_cnn", "conv1d"] else "sequence"
    print(f"\n=======================================================")
    print(f" Starting SER Training: Model='{model_type}', Feature='{feature_type}'")
    print(f" Split Strategy='{split_strategy}', Epochs={epochs}, BatchSize={batch_size}")
    print(f"=======================================================\n")
    
    # 1. Prepare splits
    split_data = prepare_dataset_splits(
        feature_type=feature_type,
        split_strategy=split_strategy
    )
    
    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    X_val = split_data["X_val"]
    y_val = split_data["y_val"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]
    scaler = split_data["scaler"]
    label_encoder = split_data["label_encoder"]
    num_classes = split_data["num_classes"]
    
    # If 1D CNN, add channel dimension: (N, D) -> (N, D, 1)
    if feature_type == "1d":
        X_train_in = np.expand_dims(X_train, axis=-1)
        X_val_in = np.expand_dims(X_val, axis=-1)
        X_test_in = np.expand_dims(X_test, axis=-1)
        input_dim_or_shape = X_train.shape[1]
    else:
        X_train_in = X_train
        X_val_in = X_val
        X_test_in = X_test
        input_dim_or_shape = (X_train.shape[1], X_train.shape[2])
        
    print(f"Training samples: {len(X_train_in)}, Val samples: {len(X_val_in)}, Test samples: {len(X_test_in)}")
    print(f"Input shape: {X_train_in.shape[1:]}, Output classes: {num_classes}")

    # 2. Save fitted scaler and label encoder
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"Saved scaler to {SCALER_PATH}")
    print(f"Saved label encoder to {LABEL_ENCODER_PATH}")

    # 3. Instantiate Model
    model = get_model(
        model_type=model_type,
        input_dim_or_shape=input_dim_or_shape,
        num_classes=num_classes,
        learning_rate=learning_rate
    )
    model.summary()

    # 4. Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # 5. Train
    history = model.fit(
        X_train_in, y_train,
        validation_data=(X_val_in, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Plot history
    plot_training_history(history)

    # 7. Evaluate on Test Set
    test_loss, test_acc = model.evaluate(X_test_in, y_test, verbose=0)
    print("\n=======================================================")
    print(f" Final Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")
    print(f" Model saved at: {BEST_MODEL_PATH}")
    print(f"=======================================================\n")

    return model, history, (test_loss, test_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Speech Emotion Recognition Model")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "lstm", "crnn"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--split", type=str, default="stratified", choices=["stratified", "speaker_independent"], help="Train/Test split strategy")
    args = parser.parse_args()

    train_pipeline(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        split_strategy=args.split
    )
