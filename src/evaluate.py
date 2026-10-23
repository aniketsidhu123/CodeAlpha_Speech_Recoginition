import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from src.config import (
    BEST_MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    CONFUSION_MATRIX_PATH,
    EVALUATION_REPORT_PATH,
    EMOTION_CLASSES
)
from src.dataset import prepare_dataset_splits
from src.models import TemporalAttention  # Needed for custom layer serialization

def evaluate_model(
    model_path=BEST_MODEL_PATH,
    scaler_path=SCALER_PATH,
    label_encoder_path=LABEL_ENCODER_PATH,
    save_cm_path=CONFUSION_MATRIX_PATH,
    save_json_path=EVALUATION_REPORT_PATH
):
    """
    Run detailed evaluation on the test set:
    - Confusion Matrix (Heatmap)
    - Classification Report (Precision, Recall, F1-score)
    - Saves plots and JSON summary
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train model first!")

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"TemporalAttention": TemporalAttention}
    )
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)

    # Determine if model expects 1D or sequence input
    input_shape = model.input_shape
    is_1d = len(input_shape) == 3 and input_shape[2] == 1
    feature_type = "1d" if is_1d else "sequence"

    print(f"Detected model input: shape={input_shape}, feature_type='{feature_type}'")
    split_data = prepare_dataset_splits(feature_type=feature_type)
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    if is_1d:
        X_test_in = np.expand_dims(X_test, axis=-1)
    else:
        X_test_in = X_test

    print(f"Running predictions on {len(X_test_in)} test samples...")
    y_pred_probs = model.predict(X_test_in)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    class_names = [label_encoder.classes_[i] for i in range(len(label_encoder.classes_))]

    clf_report_dict = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )
    clf_report_text = classification_report(
        y_test, y_pred, target_names=class_names
    )

    print("\n" + "="*60)
    print(" SPEECH EMOTION RECOGNITION - EVALUATION REPORT")
    print("="*60)
    print(f"Overall Accuracy:   {acc * 100:.2f}%")
    print(f"Macro F1-Score:     {macro_f1 * 100:.2f}%")
    print(f"Weighted F1-Score:  {weighted_f1 * 100:.2f}%")
    print("-"*60)
    print(clf_report_text)
    print("="*60)

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    
    # Combined annotation: count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)"

    ax = sns.heatmap(
        cm_normalized,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=1.0,
        linecolor="#cbd5e1"
    )
    plt.title(f"Confusion Matrix (Test Accuracy: {acc*100:.1f}%)", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Predicted Emotion", fontsize=12, labelpad=10)
    plt.ylabel("Ground Truth Emotion", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to {save_cm_path}")

    # Save JSON summary
    summary = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": clf_report_dict
    }
    with open(save_json_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Metrics saved to {save_json_path}")

    return summary

if __name__ == "__main__":
    evaluate_model()
