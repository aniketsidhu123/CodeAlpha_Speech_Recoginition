import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    BEST_MODEL_PATH,
    DEMO_SAMPLES_DIR,
    FEATURES_CACHE_PATH
)

def run_preprocess(args):
    from src.dataset import extract_and_cache_dataset
    print("Starting dataset scanning & acoustic feature extraction...")
    extract_and_cache_dataset(force_recompute=args.force)
    print("Feature extraction complete!")

def run_train(args):
    from src.train import train_pipeline
    train_pipeline(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        split_strategy=args.split
    )

def run_evaluate(args):
    from src.evaluate import evaluate_model
    evaluate_model()

def run_predict(args):
    from src.predictor import EmotionPredictor
    audio_file = Path(args.file)
    if not audio_file.exists():
        print(f"Error: File '{audio_file}' not found.")
        return

    predictor = EmotionPredictor()
    result = predictor.predict_file(audio_file)

    print("\n" + "=" * 50)
    print(f" Audio File: {result.get('filename', str(audio_file))}")
    print(f" Predicted Emotion: {result['predicted_emotion'].upper()} ({result['confidence_percent']}%)")
    print("-" * 50)
    print(" Emotion Probabilities:")
    for item in result["ranking"]:
        bar = "#" * int(item["percentage"] // 4)
        print(f"  {item['emotion'].capitalize():<12} {item['percentage']:>5.1f}% | {bar}")
    print("-" * 50)
    print(f" Duration: {result['audio_stats']['duration_sec']}s | Sample Rate: {result['audio_stats']['sample_rate']}Hz | RMS: {result['audio_stats']['rms_energy']}")
    print("=" * 50 + "\n")

def run_serve(args):
    import uvicorn
    print(f"Starting FastAPI Web Server at http://{args.host}:{args.port}")
    uvicorn.run("web.app:app", host=args.host, port=args.port, reload=args.reload)

def run_streamlit(args):
    import subprocess
    cmd = [sys.executable, "-m", "streamlit", "run", "app_streamlit.py", "--server.port", str(args.port)]
    print(f"Launching Streamlit on port {args.port}...")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(
        description="Speech Emotion Recognition (SER) CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Preprocess
    p_prep = subparsers.add_parser("preprocess", help="Extract and cache dataset features")
    p_prep.add_argument("--force", action="store_true", help="Force recomputing cached features")
    p_prep.set_defaults(func=run_preprocess)

    # Train
    p_train = subparsers.add_parser("train", help="Train deep learning model")
    p_train.add_argument("--model", type=str, default="cnn", choices=["cnn", "lstm", "crnn"], help="Model architecture")
    p_train.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    p_train.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p_train.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    p_train.add_argument("--split", type=str, default="stratified", choices=["stratified", "speaker_independent"], help="Train/test split strategy")
    p_train.set_defaults(func=run_train)

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate trained model and generate reports")
    p_eval.set_defaults(func=run_evaluate)

    # Predict
    p_pred = subparsers.add_parser("predict", help="Predict emotion on audio file")
    p_pred.add_argument("--file", type=str, required=True, help="Path to audio file (.wav, .mp3)")
    p_pred.set_defaults(func=run_predict)

    # Serve (FastAPI)
    p_serve = subparsers.add_parser("serve", help="Launch FastAPI web server")
    p_serve.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    p_serve.add_argument("--port", type=int, default=8000, help="Port number")
    p_serve.add_argument("--reload", action="store_true", help="Auto-reload on code change")
    p_serve.set_defaults(func=run_serve)

    # Streamlit
    p_st = subparsers.add_parser("streamlit", help="Launch Streamlit web dashboard")
    p_st.add_argument("--port", type=int, default=8501, help="Port number")
    p_st.set_defaults(func=run_streamlit)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)

if __name__ == "__main__":
    main()
