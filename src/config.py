import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DEMO_SAMPLES_DIR = BASE_DIR / "demo_samples"

# Ensure directories exist
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, DEMO_SAMPLES_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Dataset Resolution
DEFAULT_CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "kagglehub" / "datasets" / "uwrfkaggler" / "ravdess-emotional-speech-audio" / "versions" / "1"

def get_ravdess_path() -> Path:
    """Find the RAVDESS dataset directory."""
    env_path = os.getenv("RAVDESS_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    if RAW_DATA_DIR.exists() and any(RAW_DATA_DIR.glob("Actor_*")):
        return RAW_DATA_DIR
    if DEFAULT_CACHE_DIR.exists() and any(DEFAULT_CACHE_DIR.glob("Actor_*")):
        return DEFAULT_CACHE_DIR
    return RAW_DATA_DIR

# Audio Processing Configuration
SAMPLE_RATE = 22050
DURATION = 3.0  # target duration in seconds for fixed-size features
TOTAL_SAMPLES = int(SAMPLE_RATE * DURATION)
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
TOP_DB_TRIM = 25

# Emotion Labels Mapping for RAVDESS (Modality-Channel-Emotion-Intensity-Statement-Repetition-Actor)
# Emotion code 01 to 08
EMOTION_CODE_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

EMOTION_CLASSES = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

# Color palette for emotions in UI and plots
EMOTION_COLORS = {
    "neutral": "#94a3b8",   # Slate Gray
    "calm": "#14b8a6",      # Soft Teal
    "happy": "#f59e0b",     # Amber Gold
    "sad": "#3b82f6",       # Sapphire Blue
    "angry": "#ef4444",     # Crimson Red
    "fearful": "#8b5cf6",   # Deep Purple
    "disgust": "#10b981",   # Emerald Green
    "surprised": "#ec4899"  # Hot Pink
}

# Training Hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Saved Model Filenames
BEST_MODEL_PATH = MODELS_DIR / "best_emotion_model.keras"
SCALER_PATH = MODELS_DIR / "feature_scaler.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
FEATURES_CACHE_PATH = PROCESSED_DATA_DIR / "ravdess_extracted_features.npz"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
TRAINING_CURVES_PATH = REPORTS_DIR / "training_curves.png"
EVALUATION_REPORT_PATH = REPORTS_DIR / "classification_report.json"
