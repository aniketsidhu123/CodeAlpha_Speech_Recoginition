import os
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display

from src.config import (
    BEST_MODEL_PATH,
    DEMO_SAMPLES_DIR,
    EMOTION_COLORS,
    EMOTION_CLASSES,
    SAMPLE_RATE
)
from src.features import load_audio
from src.predictor import EmotionPredictor

# Page Setup
st.set_page_config(
    page_title="Speech Emotion Recognition (SER)",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Theme & Glassmorphism)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
    }
    
    .emotion-card {
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        background: rgba(30, 41, 59, 0.6);
        border: 2px solid #3b82f6;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }
    
    .emotion-title {
        font-size: 2.2rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
    
    .metric-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        background: rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Cache predictor in session state
@st.cache_resource
def get_cached_predictor():
    if not BEST_MODEL_PATH.exists():
        return None
    try:
        return EmotionPredictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

predictor = get_cached_predictor()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/isometric-line/100/microphone.png", width=64)
    st.title("SER System")
    st.markdown("Speech Emotion Recognition using Deep Learning (CNN / BiLSTM / CRNN) & MFCC Acoustic Features.")
    
    st.markdown("---")
    st.subheader("Model Status")
    if BEST_MODEL_PATH.exists():
        st.success(" Trained Model Loaded")
        st.caption(f"Path: `{BEST_MODEL_PATH.name}`")
    else:
        st.warning("⚠️ No trained model found")
        st.info("Run `python main.py train` to train the deep learning model.")

    st.markdown("---")
    st.subheader("Recognized Emotions")
    for emo in EMOTION_CLASSES:
        color = EMOTION_COLORS.get(emo, "#fff")
        st.markdown(
            f"<div style='display:flex;align-items:center;margin-bottom:6px;'>"
            f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;background:{color};margin-right:8px;'></span>"
            f"<span style='text-transform:capitalize;'>{emo}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

# Main Header
st.markdown("""
<div class="hero-container">
    <h1 style="margin:0 0 10px 0; font-size: 2.4rem; font-weight:800; background: linear-gradient(90deg, #60a5fa, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        🎙️ Speech Emotion Recognition (SER)
    </h1>
    <p style="margin:0; font-size: 1.1rem; color: #94a3b8; max-width: 800px;">
        Deep learning speech signal processing system to detect human emotional states from voice acoustic characteristics, trained on the RAVDESS dataset using MFCCs, Chroma, and Mel-Spectrogram features.
    </p>
</div>
""", unsafe_allow_html=True)

if predictor is None:
    st.warning("Please train the model first by running `python main.py train` or wait for training to finish.")
    st.stop()

# Input Options
col_input1, col_input2 = st.columns([1, 1], gap="medium")

selected_audio_path = None
uploaded_audio_bytes = None

with col_input1:
    st.subheader("1. Provide Audio Input")
    input_tab1, input_tab2 = st.tabs(["Demo Samples (RAVDESS)", "Upload Audio File"])
    
    with input_tab1:
        st.caption("Select a pre-loaded sample from the RAVDESS emotional speech dataset:")
        sample_files = list(DEMO_SAMPLES_DIR.glob("*.wav"))
        if sample_files:
            demo_names = [f.stem.capitalize() for f in sample_files]
            chosen_demo = st.selectbox("Choose Emotion Sample:", demo_names)
            selected_audio_path = DEMO_SAMPLES_DIR / f"{chosen_demo.lower()}.wav"
        else:
            st.info("No demo samples found in demo_samples directory.")
            
    with input_tab2:
        uploaded_file = st.file_uploader("Upload .wav or .mp3 voice recording:", type=["wav", "mp3", "ogg"])
        if uploaded_file is not None:
            uploaded_audio_bytes = uploaded_file.read()
            selected_audio_path = None

with col_input2:
    st.subheader("2. Audio Playback & Signal")
    if selected_audio_path and selected_audio_path.exists():
        st.audio(str(selected_audio_path), format="audio/wav")
        audio_to_analyze = selected_audio_path
    elif uploaded_audio_bytes:
        st.audio(uploaded_audio_bytes, format="audio/wav")
        audio_to_analyze = uploaded_audio_bytes
    else:
        st.info("Select a demo sample or upload a voice file to begin recognition.")
        audio_to_analyze = None

if audio_to_analyze:
    st.markdown("---")
    
    # Run Prediction
    with st.spinner("Analyzing acoustic features (MFCC, Mel-spec, Chroma) & predicting emotion..."):
        if isinstance(audio_to_analyze, Path):
            result = predictor.predict_file(audio_to_analyze)
            y, sr = load_audio(str(audio_to_analyze), target_duration=None)
        else:
            result = predictor.predict_bytes(audio_to_analyze)
            import soundfile as sf, io
            y, sr = sf.read(io.BytesIO(audio_to_analyze))
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)

    # Display Top Emotion Card
    top_emo = result["predicted_emotion"]
    top_conf = result["confidence_percent"]
    top_color = result["color"]

    st.markdown(f"""
    <div class="emotion-card" style="border-color: {top_color};">
        <span class="metric-badge" style="color: {top_color}; border: 1px solid {top_color};">Predicted Emotion</span>
        <div class="emotion-title" style="color: {top_color};">{top_emo}</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #f8fafc;">Confidence: {top_conf}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Two Column Layout: Probabilities Bar Chart & Audio Plots
    col_chart, col_plots = st.columns([1, 1], gap="medium")

    with col_chart:
        st.subheader("Emotion Probability Distribution")
        probs_df = pd.DataFrame(result["ranking"])
        
        # Horizontal bar chart using matplotlib
        fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
        fig_bar.patch.set_facecolor('#0f172a')
        ax_bar.set_facecolor('#1e293b')
        
        y_pos = np.arange(len(probs_df))
        bars = ax_bar.barh(
            y_pos,
            probs_df["probability"] * 100,
            color=probs_df["color"],
            height=0.6,
            edgecolor="none"
        )
        
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels([e.capitalize() for e in probs_df["emotion"]], color="#f8fafc", fontsize=11, fontweight="600")
        ax_bar.set_xlabel("Probability (%)", color="#94a3b8", fontsize=11)
        ax_bar.set_xlim(0, 100)
        ax_bar.invert_yaxis()  # Top emotion first
        ax_bar.tick_params(colors="#94a3b8")
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.spines['bottom'].set_color('#334155')
        ax_bar.spines['left'].set_color('#334155')
        ax_bar.grid(axis='x', linestyle=':', alpha=0.3, color="#94a3b8")

        for bar in bars:
            width = bar.get_width()
            if width > 3:
                ax_bar.text(
                    width + 1.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.1f}%",
                    va='center',
                    color="#f8fafc",
                    fontweight="600",
                    fontsize=10
                )

        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

        # Audio Signal Metrics
        st.subheader("Audio Signal Metrics")
        stats = result["audio_stats"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Duration", f"{stats['duration_sec']} s")
        m2.metric("Sample Rate", f"{stats['sample_rate']} Hz")
        m3.metric("RMS Energy", f"{stats['rms_energy']}")
        m4.metric("Zero Crossing", f"{stats['zero_crossing_rate']}")

    with col_plots:
        st.subheader("Audio Waveform & Spectrogram")
        
        # Waveform
        fig_wave, ax_wave = plt.subplots(figsize=(8, 2.2))
        fig_wave.patch.set_facecolor('#0f172a')
        ax_wave.set_facecolor('#1e293b')
        librosa.display.waveshow(y, sr=sr, ax=ax_wave, color="#38bdf8", alpha=0.85)
        ax_wave.set_title("Waveform (Time Domain)", color="#f8fafc", fontsize=11, fontweight="bold")
        ax_wave.tick_params(colors="#94a3b8")
        ax_wave.spines['top'].set_visible(False)
        ax_wave.spines['right'].set_visible(False)
        ax_wave.spines['bottom'].set_color('#334155')
        ax_wave.spines['left'].set_color('#334155')
        plt.tight_layout()
        st.pyplot(fig_wave)
        plt.close(fig_wave)

        # Mel-Spectrogram
        fig_mel, ax_mel = plt.subplots(figsize=(8, 2.5))
        fig_mel.patch.set_facecolor('#0f172a')
        ax_mel.set_facecolor('#1e293b')
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        img = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=ax_mel, cmap='magma')
        ax_mel.set_title("Log Mel-Spectrogram (Frequency Domain)", color="#f8fafc", fontsize=11, fontweight="bold")
        ax_mel.tick_params(colors="#94a3b8")
        ax_mel.spines['top'].set_visible(False)
        ax_mel.spines['right'].set_visible(False)
        ax_mel.spines['bottom'].set_color('#334155')
        ax_mel.spines['left'].set_color('#334155')
        plt.tight_layout()
        st.pyplot(fig_mel)
        plt.close(fig_mel)
