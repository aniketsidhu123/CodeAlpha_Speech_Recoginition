# 🎙️ Speech Emotion Recognition (SER) System

A deep learning speech signal processing project to recognize human emotional states from voice recordings using **MFCCs (Mel-Frequency Cepstral Coefficients)**, **Mel-Spectrograms**, and deep neural networks (**1D CNN**, **BiLSTM**, and **Hybrid CRNN with Self-Attention**), trained on the **RAVDESS** dataset.

---

## 📌 Project Overview & Objectives

Human speech contains rich emotional nuances embedded in pitch contours, spectral energy distribution, harmonic characteristics, and temporal speech rhythm. 

This project implements an end-to-end Speech Emotion Recognition pipeline:
1. **Speech Signal Processing**: Extracts multi-perspective acoustic features (MFCCs, Mel-Spectrograms, Chroma STFT, Spectral Contrast, Tonnetz, Zero-Crossing Rate, and RMS energy).
2. **Deep Learning Architectures**:
   - **Deep 1D CNN**: Convolutional filters capturing multi-scale spectral patterns.
   - **Bidirectional LSTM (BiLSTM)**: Recurrent neural network capturing forward and backward temporal phonetic transitions.
   - **Hybrid CRNN with Self-Attention**: Convolutional frontend for local acoustic representation, Bidirectional LSTM for context modeling, and Bahdanau-style Attention pooling to weigh emotionally expressive speech segments.
3. **Interactive User Interfaces**:
   - **FastAPI + Modern Dark Glassmorphic Web App**: In-browser microphone recording, audio waveform visualizer, drag-and-drop file upload, real-time emotion probability bars, and audio signal metrics.
   - **Streamlit Interactive Dashboard**: Side-by-side waveform/spectrogram exploration and emotion classification.
4. **Automated One-Click Scripts**: `setup.bat` and `run.bat` for Windows execution.

---

## 🎯 Recognized Emotions

The system classifies speech into **8 distinct emotional categories** from the RAVDESS dataset:

| Emotion | Description | Accent Color |
| :--- | :--- | :--- |
| **Happy** | Elevated pitch, higher energy, expanded pitch range | Amber Gold (`#f59e0b`) |
| **Sad** | Low acoustic energy, falling pitch contours, slower pace | Sapphire Blue (`#3b82f6`) |
| **Angry** | High RMS energy, sharp spectral attack, elevated pitch | Crimson Red (`#ef4444`) |
| **Calm** | Steady tempo, smooth pitch slope, relaxed articulation | Soft Teal (`#14b8a6`) |
| **Fearful** | Rapid tempo, jitter, higher fundamental frequency | Deep Purple (`#8b5cf6`) |
| **Disgust** | Lower average pitch, nasal acoustic coloration | Emerald Green (`#10b981`) |
| **Surprised** | Sudden pitch shift, high frequency energy bursts | Hot Pink (`#ec4899`) |
| **Neutral** | Baseline prosody and energy distribution | Slate Gray (`#94a3b8`) |

---

## 📊 Acoustic Feature Engineering

The feature extraction engine (`src/features.py`) extracts a **289-dimensional acoustic feature vector**:

| Feature Type | Dimensions | Acoustic Significance |
| :--- | :---: | :--- |
| **MFCCs** | 120 | 40 coefficients (Mean, Std, Delta mean) representing vocal tract timbre |
| **Chroma STFT** | 24 | 12 pitch classes (Mean, Std) capturing harmonic distribution |
| **Mel-Spectrogram** | 128 | 128 log-mel frequency bands representing psychoacoustic perception |
| **Spectral Contrast**| 7 | Contrast between spectral peaks and valleys (harmonic clarity) |
| **Tonnetz** | 6 | Tonal centroid features capturing harmonic pitch space transitions |
| **Zero Crossing Rate**| 2 | Rate of sign-changes (Mean, Std) distinguishing voiced/unvoiced sounds |
| **RMS Energy** | 2 | Root-mean-square amplitude (Mean, Std) representing vocal intensity |

For recurrent and temporal models (`BiLSTM` and `CRNN`), time-aligned 2D feature matrices of shape `(130, 93)` (Time-steps × Spectro-Temporal Features) are extracted.

---

## 📁 Repository Directory Structure

```
d:/Emotion_Recognition_CA/
├── setup.bat                # 1-click environment setup and dependency installer
├── run.bat                  # Interactive control center & menu launcher
├── main.py                  # Unified CLI for preprocess, train, eval, predict, serve
├── requirements.txt         # Project dependencies
├── app_streamlit.py         # Streamlit interactive dashboard
├── demo_samples/            # Curated RAVDESS audio samples for all 8 emotions
│   ├── happy.wav
│   ├── angry.wav
│   ├── sad.wav
│   ├── calm.wav
│   ├── fearful.wav
│   ├── disgust.wav
│   ├── surprised.wav
│   └── neutral.wav
├── data/
│   ├── raw/                 # RAVDESS dataset audio files
│   └── processed/           # Cached extracted features (.npz)
├── models/                  # Trained neural networks, scalers, and label encoders
│   ├── best_emotion_model.keras
│   ├── feature_scaler.joblib
│   └── label_encoder.joblib
├── reports/                 # Confusion matrix and evaluation reports
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   └── classification_report.json
├── src/
│   ├── config.py            # Central parameters, paths, emotion maps, colors
│   ├── dataset.py           # RAVDESS parser, threading feature cache, splits
│   ├── features.py          # Acoustic signal processing, MFCC/Mel/Chroma extraction
│   ├── models.py            # 1D CNN, BiLSTM, CRNN with Attention architectures
│   ├── train.py             # Training loop, callbacks, early stopping, history plot
│   ├── evaluate.py          # Test set evaluation, metrics, heatmap generator
│   └── predictor.py         # Inference engine for files and raw audio streams
└── web/
    ├── app.py               # FastAPI backend with audio endpoints
    ├── templates/
    │   └── index.html       # Single-page glassmorphism application
    └── static/
        ├── css/style.css    # Responsive dark-mode styling and animations
        └── js/app.js        # Web Audio API recording, waveform, dynamic UI
```

---

## 🚀 Quickstart Guide

### Option 1: Automatic Windows Setup (`setup.bat` & `run.bat`)

1. **Run Setup**: Double-click `setup.bat` (or run in cmd):
   ```cmd
   setup.bat
   ```
   *Verifies Python, creates directories, sets up venv, and installs dependencies.*

2. **Launch Control Center**: Double-click `run.bat`:
   ```cmd
   run.bat
   ```
   *Select from the interactive menu to train models, launch the web UI, or evaluate.*

### Option 2: Command Line (CLI)

```bash
# 1. Extract and cache features
python main.py preprocess

# 2. Train deep learning model (options: cnn, lstm, crnn)
python main.py train --model cnn --epochs 40 --batch-size 32

# 3. Evaluate trained model and generate Confusion Matrix
python main.py evaluate

# 4. Predict emotion on any audio file
python main.py predict --file demo_samples/happy.wav

# 5. Launch FastAPI Modern Web Interface
python main.py serve --port 8000

# 6. Launch Streamlit Dashboard
python main.py streamlit --port 8501
```

---

## 📈 Model Architectures

### 1. Deep 1D CNN (`CNN_1D_SER`)
- Input: `(289, 1)` Normalized acoustic feature vector
- 4x Conv1D blocks (64, 128, 256, 128 filters) with Batch Normalization, ReLU activations, MaxPooling, and Spatial Dropout
- Global Average Pooling 1D + Dense(128, L2 regularized) + Dropout(0.4)
- Softmax output across 8 emotion classes

### 2. Bidirectional LSTM (`BiLSTM_SER`)
- Input: `(130, 93)` Time-series spectral sequence
- Bidirectional LSTM (128 units, dropout=0.2) + Batch Normalization
- Bidirectional LSTM (64 units, dropout=0.2)
- Dense(64) + Dropout(0.3) + Softmax output

### 3. Hybrid CRNN with Self-Attention (`CRNN_Attention_SER`)
- Convolutional frontend capturing localized spectral patterns
- Bidirectional LSTM middle-end modeling temporal emotional dynamics
- Custom Bahdanau-style **Temporal Attention Layer** learning adaptive weights over speech frames:
  $$u_t = \tanh(W h_t + b)$$
  $$\alpha_t = \frac{\exp(u_t^\top u)}{\sum_t \exp(u_t^\top u)}$$
  $$v = \sum_t \alpha_t h_t$$
- Dense classifier producing calibrated class probabilities

---

## 🧪 Evaluation & Results

Evaluation generates:
- **Confusion Matrix Heatmap** (`reports/confusion_matrix.png`): True vs. predicted counts and row-normalized percentages.
- **Accuracy & Loss Curves** (`reports/training_curves.png`): Epoch-by-epoch training and validation metrics.
- **Classification Report** (`reports/classification_report.json`): Per-class Precision, Recall, and F1-score.
