import os
import io
import joblib
import numpy as np
import soundfile as sf
import tensorflow as tf
from typing import Dict, Any, Union, Optional
from pathlib import Path

from src.config import (
    BEST_MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    SAMPLE_RATE,
    EMOTION_COLORS
)
from src.features import (
    load_audio,
    extract_features_1d,
    extract_sequence_features,
    get_audio_signal_stats
)
from src.models import TemporalAttention

class EmotionPredictor:
    """
    Inference class for recognizing emotions from raw audio files or audio arrays.
    """
    def __init__(
        self,
        model_path: Union[str, Path] = BEST_MODEL_PATH,
        scaler_path: Union[str, Path] = SCALER_PATH,
        label_encoder_path: Union[str, Path] = LABEL_ENCODER_PATH
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.label_encoder_path = Path(label_encoder_path)
        
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.is_1d_model = True
        
        self._load_artifacts()

    def _load_artifacts(self):
        """Load trained neural net, scaler, and label encoder."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Train the model first."
            )
        if not self.scaler_path.exists() or not self.label_encoder_path.exists():
            raise FileNotFoundError(
                f"Preprocessors missing: Scaler={self.scaler_path.exists()}, Labels={self.label_encoder_path.exists()}"
            )

        self.model = tf.keras.models.load_model(
            str(self.model_path),
            custom_objects={"TemporalAttention": TemporalAttention}
        )
        self.scaler = joblib.load(self.scaler_path)
        self.label_encoder = joblib.load(self.label_encoder_path)
        
        # Determine model input dimensionality
        shape = self.model.input_shape
        self.is_1d_model = len(shape) == 3 and shape[2] == 1

    def predict_audio(self, y: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, Any]:
        """
        Predict emotion from audio waveform array (y, sr).
        """
        stats = get_audio_signal_stats(y, sr)
        
        if self.is_1d_model:
            feat = extract_features_1d(y, sr=sr)
            feat_scaled = self.scaler.transform(feat.reshape(1, -1))
            model_input = np.expand_dims(feat_scaled, axis=-1)  # (1, 289, 1)
        else:
            feat_seq = extract_sequence_features(y, sr=sr)  # (130, 93)
            shape = feat_seq.shape
            feat_2d = feat_seq.reshape(-1, shape[-1])
            feat_scaled = self.scaler.transform(feat_2d).reshape(1, *shape)
            model_input = feat_scaled

        # Inference
        probs = self.model.predict(model_input, verbose=0)[0]
        classes = self.label_encoder.classes_
        
        # Sort predictions
        class_probs = {classes[i]: float(probs[i]) for i in range(len(classes))}
        sorted_emotions = sorted(class_probs.items(), key=lambda item: item[1], reverse=True)
        
        top_emotion, top_confidence = sorted_emotions[0]
        
        # Color mapping
        top_color = EMOTION_COLORS.get(top_emotion, "#6366f1")

        return {
            "predicted_emotion": top_emotion,
            "confidence": round(top_confidence, 4),
            "confidence_percent": round(top_confidence * 100, 1),
            "color": top_color,
            "probabilities": {k: round(v, 4) for k, v in class_probs.items()},
            "ranking": [
                {
                    "emotion": emo,
                    "probability": round(p, 4),
                    "percentage": round(p * 100, 1),
                    "color": EMOTION_COLORS.get(emo, "#94a3b8")
                }
                for emo, p in sorted_emotions
            ],
            "audio_stats": stats
        }

    def predict_file(self, audio_file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load audio from file and predict."""
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        y, sr = load_audio(str(audio_path), sr=SAMPLE_RATE)
        result = self.predict_audio(y, sr=sr)
        result["filename"] = audio_path.name
        return result

    def predict_bytes(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Predict emotion directly from uploaded audio bytes."""
        bio = io.BytesIO(audio_bytes)
        try:
            y, sr = sf.read(bio)
        except Exception as e_sf:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                import librosa
                y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE)
            except Exception as e_lib:
                raise ValueError(
                    f"Unsupported audio encoding. Please record or upload standard WAV/MP3 audio. (Error: {e_sf})"
                )
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
        
        # If stereo, average to mono
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
            
        # Resample if needed
        if sr != SAMPLE_RATE:
            import librosa
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
            
        result = self.predict_audio(y, sr=sr)
        return result
