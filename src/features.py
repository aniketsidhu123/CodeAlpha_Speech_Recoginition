import numpy as np
import soundfile as sf
import librosa
from typing import Tuple, Dict, Any, Optional
from src.config import SAMPLE_RATE, DURATION, TOTAL_SAMPLES, N_MFCC, N_MELS, HOP_LENGTH, N_FFT, TOP_DB_TRIM

# ==============================================================================
# Audio Augmentation Functions
# ==============================================================================

def add_noise(y: np.ndarray, noise_factor: float = 0.015) -> np.ndarray:
    """Add Gaussian white noise to the audio signal."""
    noise = np.random.randn(len(y))
    augmented = y + noise_factor * np.amax(y) * noise
    return augmented.astype(np.float32)

def pitch_shift(y: np.ndarray, sr: int = SAMPLE_RATE, n_steps: float = 2.0) -> np.ndarray:
    """Shift the pitch of the audio signal up or down."""
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

def time_stretch(y: np.ndarray, rate: float = 1.1) -> np.ndarray:
    """Stretch or compress the time duration of the audio without changing pitch."""
    return librosa.effects.time_stretch(y=y, rate=rate)

def time_shift(y: np.ndarray, shift_factor: float = 0.2) -> np.ndarray:
    """Roll/shift audio samples horizontally in time."""
    shift = int(shift_factor * len(y))
    return np.roll(y, shift)

# ==============================================================================
# Pure NumPy Robust Signal Utilities (Crash-Free, no Numba bugs)
# ==============================================================================

def compute_zcr_pure_numpy(y: np.ndarray, frame_length: int = N_FFT, hop_length: int = HOP_LENGTH) -> Tuple[float, float]:
    """Compute Zero Crossing Rate using pure NumPy without Numba gufunc."""
    pad_width = frame_length // 2
    y_padded = np.pad(y, pad_width, mode='reflect')
    frames = np.lib.stride_tricks.sliding_window_view(y_padded, frame_length)[::hop_length]
    
    signs = np.sign(frames)
    signs[signs == 0] = 1
    crossings = np.abs(np.diff(signs, axis=-1)) / 2.0
    zcr_per_frame = np.mean(crossings, axis=-1)
    return float(np.mean(zcr_per_frame)), float(np.std(zcr_per_frame))

def compute_rms_pure_numpy(y: np.ndarray, frame_length: int = N_FFT, hop_length: int = HOP_LENGTH) -> Tuple[np.ndarray, float, float]:
    """Compute RMS energy per frame and summary stats using pure NumPy."""
    pad_width = frame_length // 2
    y_padded = np.pad(y, pad_width, mode='reflect')
    frames = np.lib.stride_tricks.sliding_window_view(y_padded, frame_length)[::hop_length]
    rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=-1) + 1e-10)
    return rms_per_frame, float(np.mean(rms_per_frame)), float(np.std(rms_per_frame))

# ==============================================================================
# Audio Loading & Preprocessing
# ==============================================================================

def load_audio(
    file_path: str,
    sr: int = SAMPLE_RATE,
    trim_silence: bool = True,
    target_duration: Optional[float] = DURATION
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, optionally trim silence, and adjust length to target_duration.
    Uses fast soundfile C backend with librosa fallback.
    """
    try:
        y, sample_rate = sf.read(file_path)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        if sample_rate != sr:
            y = librosa.resample(y.astype(np.float32), orig_sr=sample_rate, target_sr=sr)
            sample_rate = sr
        else:
            y = y.astype(np.float32)
    except Exception:
        y, sample_rate = librosa.load(file_path, sr=sr)
    
    if trim_silence and len(y) > 0:
        y_trimmed, _ = librosa.effects.trim(y, top_db=TOP_DB_TRIM)
        # Avoid trimming to empty
        if len(y_trimmed) > sample_rate * 0.3:
            y = y_trimmed
            
    if target_duration is not None:
        target_len = int(target_duration * sample_rate)
        if len(y) < target_len:
            pad_width = target_len - len(y)
            y = np.pad(y, (0, pad_width), mode='constant')
        else:
            y = y[:target_len]
            
    return y, sample_rate

# ==============================================================================
# Feature Extraction: 1D Aggregated Feature Vector
# ==============================================================================

def extract_features_1d(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract a comprehensive 1D acoustic feature vector:
    - MFCCs (mean, std, delta mean) -> 40 + 40 + 40 = 120
    - Chroma STFT (mean, std) -> 12 + 12 = 24
    - Mel-spectrogram (mean) -> 128
    - Spectral Contrast (mean) -> 7
    - Tonnetz (mean) -> 6
    - Zero Crossing Rate (mean, std) -> 2
    - RMS Energy (mean, std) -> 2
    Total: 289 features
    """
    features = []
    
    # 1. MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.append(np.mean(mfcc.T, axis=0))  # 40
    features.append(np.std(mfcc.T, axis=0))   # 40
    mfcc_delta = librosa.feature.delta(mfcc)
    features.append(np.mean(mfcc_delta.T, axis=0))  # 40
    
    # 2. Chroma STFT
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
    features.append(np.mean(chroma.T, axis=0))  # 12
    features.append(np.std(chroma.T, axis=0))   # 12
    
    # 3. Mel Spectrogram (log scale)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.append(np.mean(mel_db.T, axis=0))  # 128
    
    # 4. Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LENGTH)
    features.append(np.mean(contrast.T, axis=0))  # 7
    
    # 5. Tonnetz (Tonal Centroid) - Fast computation from chroma
    try:
        tonnetz = librosa.feature.tonnetz(chroma=chroma)
        features.append(np.mean(tonnetz.T, axis=0))  # 6
    except Exception:
        features.append(np.zeros(6, dtype=np.float32))
    
    # 6. Zero Crossing Rate (Pure NumPy)
    zcr_mean, zcr_std = compute_zcr_pure_numpy(y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    features.append(np.array([zcr_mean, zcr_std]))  # 2
    
    # 7. RMS Energy (Pure NumPy)
    _, rms_mean, rms_std = compute_rms_pure_numpy(y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    features.append(np.array([rms_mean, rms_std]))  # 2
    
    return np.hstack(features).astype(np.float32)

# ==============================================================================
# Feature Extraction: 2D Sequence Matrix (Time x Features) for LSTM/CRNN
# ==============================================================================

def extract_sequence_features(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    max_len: int = 130
) -> np.ndarray:
    """
    Extract a sequence representation (TimeSteps x Features) for RNN / LSTM / CRNN.
    Concatenates:
    - MFCC (40)
    - Delta MFCC (40)
    - Chroma (12)
    - RMS (1)
    Total features per timestep: 93
    Padded or cropped to `max_len` timesteps.
    Output shape: (max_len, 93)
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    mfcc_delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
    rms_frames, _, _ = compute_rms_pure_numpy(y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    
    # Align RMS frame count with STFT frames
    target_frames = mfcc.shape[1]
    if len(rms_frames) < target_frames:
        rms_frames = np.pad(rms_frames, (0, target_frames - len(rms_frames)), mode='edge')
    else:
        rms_frames = rms_frames[:target_frames]
    rms_frames = rms_frames.reshape(1, target_frames)
    
    # Stack: (40 + 40 + 12 + 1, time_steps) = (93, time_steps)
    combined = np.vstack([mfcc, mfcc_delta, chroma, rms_frames])  # shape: (93, T)
    seq = combined.T  # shape: (T, 93)
    
    # Pad or truncate to fixed length
    if seq.shape[0] < max_len:
        pad_width = max_len - seq.shape[0]
        seq = np.pad(seq, ((0, pad_width), (0, 0)), mode='constant')
    else:
        seq = seq[:max_len, :]
        
    return seq.astype(np.float32)

# ==============================================================================
# Audio Visual Feature Inspection Helper
# ==============================================================================

def get_audio_signal_stats(y: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, Any]:
    """Compute high-level signal statistics for UI / display."""
    _, rms_mean, _ = compute_rms_pure_numpy(y)
    zcr_mean, _ = compute_zcr_pure_numpy(y)
    duration = float(len(y) / sr)
    
    return {
        "duration_sec": round(duration, 2),
        "sample_rate": sr,
        "rms_energy": round(rms_mean, 4),
        "zero_crossing_rate": round(zcr_mean, 4)
    }
