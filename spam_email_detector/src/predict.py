"""
predict.py

Load the trained model + vectorizer and classify new email text.

Usage:
    python src/predict.py "Subject line here" "Body text here"
    python src/predict.py --file path/to/email.txt
    python src/predict.py   (interactive mode, just paste text and hit Enter)
"""

import argparse
import sys
from pathlib import Path

import joblib

from preprocessing import preprocess

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_artifacts():
    model_path = MODELS_DIR / "spam_classifier.joblib"
    vec_path = MODELS_DIR / "tfidf_vectorizer.joblib"

    if not model_path.exists() or not vec_path.exists():
        sys.exit(
            "No trained model found in models/. Run `python src/train.py` first."
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


def classify(text: str, model, vectorizer) -> dict:
    cleaned = preprocess(text)
    vec = vectorizer.transform([cleaned])

    pred = model.predict(vec)[0]
    label = "SPAM" if pred == 1 else "HAM"

    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(vec)[0][pred]
    else:
        confidence = None

    return {"label": label, "confidence": confidence}


def main():
    parser = argparse.ArgumentParser(description="Classify an email as spam or ham.")
    parser.add_argument("subject", nargs="?", default="", help="Email subject")
    parser.add_argument("body", nargs="?", default="", help="Email body")
    parser.add_argument("--file", help="Path to a .txt file containing the email")
    args = parser.parse_args()

    model, vectorizer = load_artifacts()

    if args.file:
        text = Path(args.file).read_text(errors="ignore")
    elif args.subject or args.body:
        text = f"{args.subject} {args.body}"
    else:
        print("Paste email text (Ctrl+D to finish):")
        text = sys.stdin.read()

    if not text.strip():
        sys.exit("No text provided.")

    result = classify(text, model, vectorizer)

    print(f"\nPrediction: {result['label']}")
    if result["confidence"] is not None:
        print(f"Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()
