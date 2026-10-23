import os
import re
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import (
    get_ravdess_path,
    EMOTION_CODE_MAP,
    EMOTION_CLASSES,
    FEATURES_CACHE_PATH,
    RANDOM_SEED,
    VAL_SPLIT,
    TEST_SPLIT
)
from src.features import (
    load_audio,
    extract_features_1d,
    extract_sequence_features,
    add_noise,
    pitch_shift
)

# ==============================================================================
# Dataset Parsing
# ==============================================================================

def scan_ravdess_files(dataset_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Scan RAVDESS dataset directory and parse metadata from filenames.
    Filename format: 03-01-06-01-02-01-12.wav
    Parts:
      0: Modality (01=full-AV, 02=video-only, 03=audio-only)
      1: Vocal channel (01=speech, 02=song)
      2: Emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)
      3: Emotional intensity (01=normal, 02=strong)
      4: Statement (01="Kids are talking by the door", 02="Dogs are sitting by the door")
      5: Repetition (01=1st repetition, 02=2nd repetition)
      6: Actor (01 to 24. Odd = male, Even = female)
    """
    if dataset_path is None:
        dataset_path = get_ravdess_path()

    records = []
    wav_files = list(Path(dataset_path).glob("**/*.wav"))
    
    for file_path in wav_files:
        filename = file_path.name
        stem = file_path.stem
        parts = stem.split("-")
        
        if len(parts) >= 7:
            emotion_code = parts[2]
            emotion = EMOTION_CODE_MAP.get(emotion_code, None)
            if emotion is None:
                continue
                
            intensity_code = parts[3]
            actor_id = int(parts[6])
            gender = "female" if actor_id % 2 == 0 else "male"
            channel = "speech" if parts[1] == "01" else "song"
            
            records.append({
                "file_path": str(file_path),
                "filename": filename,
                "emotion": emotion,
                "intensity": "strong" if intensity_code == "02" else "normal",
                "actor_id": actor_id,
                "gender": gender,
                "channel": channel
            })
            
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset=["filename"]).reset_index(drop=True)
    return df

# ==============================================================================
# Feature Extraction Worker
# ==============================================================================

def _process_single_audio(file_path: str, emotion: str, actor_id: int, gender: str) -> Optional[Dict[str, Any]]:
    """Worker function to process one audio file."""
    try:
        y, sr = load_audio(file_path)
        feat_1d = extract_features_1d(y, sr=sr)
        feat_seq = extract_sequence_features(y, sr=sr)
        
        return {
            "feat_1d": feat_1d,
            "feat_seq": feat_seq,
            "emotion": emotion,
            "actor_id": actor_id,
            "gender": gender
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ==============================================================================
# Extract and Cache Dataset Features
# ==============================================================================

def extract_and_cache_dataset(
    force_recompute: bool = False,
    n_jobs: int = -1,
    max_samples: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Extract features for all RAVDESS audio files and cache them to disk.
    If cache exists and force_recompute is False, load directly from cache.
    """
    if FEATURES_CACHE_PATH.exists() and not force_recompute:
        print(f"Loading cached features from {FEATURES_CACHE_PATH}...")
        data = np.load(FEATURES_CACHE_PATH, allow_pickle=True)
        return {
            "X_1d": data["X_1d"],
            "X_seq": data["X_seq"],
            "y": data["y"],
            "actor_id": data["actor_id"],
            "gender": data["gender"]
        }

    df = scan_ravdess_files()
    if df.empty:
        raise FileNotFoundError(
            f"No RAVDESS audio files found. Checked: {get_ravdess_path()}"
        )
    
    if max_samples is not None:
        df = df.sample(n=min(max_samples, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"Found {len(df)} audio files across {df['actor_id'].nunique()} actors.")
    from concurrent.futures import ThreadPoolExecutor

    workers = 8
    print(f"Extracting acoustic features using {workers} concurrent threads...")

    tasks = [
        (row["file_path"], row["emotion"], row["actor_id"], row["gender"])
        for _, row in df.iterrows()
    ]

    valid_results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_single_audio, *t) for t in tasks]
        for f in tqdm(futures, total=len(futures), desc="Extracting Features"):
            res = f.result()
            if res is not None:
                valid_results.append(res)
    
    X_1d = np.array([r["feat_1d"] for r in valid_results], dtype=np.float32)
    X_seq = np.array([r["feat_seq"] for r in valid_results], dtype=np.float32)
    y = np.array([r["emotion"] for r in valid_results])
    actor_id = np.array([r["actor_id"] for r in valid_results], dtype=np.int32)
    gender = np.array([r["gender"] for r in valid_results])

    print(f"Extracted features shape: X_1d={X_1d.shape}, X_seq={X_seq.shape}, y={y.shape}")

    # Cache to disk
    np.savez_compressed(
        FEATURES_CACHE_PATH,
        X_1d=X_1d,
        X_seq=X_seq,
        y=y,
        actor_id=actor_id,
        gender=gender
    )
    print(f"Saved extracted features to {FEATURES_CACHE_PATH}")

    return {
        "X_1d": X_1d,
        "X_seq": X_seq,
        "y": y,
        "actor_id": actor_id,
        "gender": gender
    }

# ==============================================================================
# Train / Val / Test Splitting
# ==============================================================================

def prepare_dataset_splits(
    feature_type: str = "1d",
    split_strategy: str = "stratified",  # 'stratified' or 'speaker_independent'
    test_actors: Optional[List[int]] = None,
    val_actors: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Prepare normalized train, validation, and test splits with fitted scaler and label encoder.
    """
    data = extract_and_cache_dataset()
    X = data["X_1d"] if feature_type == "1d" else data["X_seq"]
    y_raw = data["y"]
    actors = data["actor_id"]

    # Encode labels
    label_encoder = LabelEncoder()
    # Ensure all EMOTION_CLASSES are in encoder classes
    label_encoder.fit(EMOTION_CLASSES)
    y_encoded = label_encoder.transform(y_raw)

    if split_strategy == "speaker_independent":
        # Actor 23 (male) and 24 (female) for test; 21, 22 for validation
        if test_actors is None:
            test_actors = [23, 24]
        if val_actors is None:
            val_actors = [21, 22]

        test_mask = np.isin(actors, test_actors)
        val_mask = np.isin(actors, val_actors)
        train_mask = ~(test_mask | val_mask)

        X_train, y_train = X[train_mask], y_encoded[train_mask]
        X_val, y_val = X[val_mask], y_encoded[val_mask]
        X_test, y_test = X[test_mask], y_encoded[test_mask]
    else:
        # Stratified train/val/test split
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y_encoded,
            test_size=TEST_SPLIT,
            random_state=RANDOM_SEED,
            stratify=y_encoded
        )
        val_relative_size = VAL_SPLIT / (1.0 - TEST_SPLIT)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_relative_size,
            random_state=RANDOM_SEED,
            stratify=y_train_val
        )

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    if feature_type == "1d":
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
    else:
        # 3D Sequence: scale across timesteps
        orig_shape_train = X_train.shape
        orig_shape_val = X_val.shape
        orig_shape_test = X_test.shape
        
        X_train_2d = X_train.reshape(-1, orig_shape_train[-1])
        scaler.fit(X_train_2d)
        
        X_train_scaled = scaler.transform(X_train_2d).reshape(orig_shape_train)
        X_val_scaled = scaler.transform(X_val.reshape(-1, orig_shape_val[-1])).reshape(orig_shape_val)
        X_test_scaled = scaler.transform(X_test.reshape(-1, orig_shape_test[-1])).reshape(orig_shape_test)

    return {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "num_classes": len(EMOTION_CLASSES)
    }
