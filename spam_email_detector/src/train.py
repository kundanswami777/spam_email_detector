"""
train.py

End-to-end training pipeline:
  1. Load + clean the Enron dataset (preprocessing.py)
  2. TF-IDF vectorize
  3. GridSearchCV (5-fold) over Naive Bayes, Linear SVM, and XGBoost
  4. Pick the best model by F1, evaluate it on the held-out test set
  5. Save the winning model + vectorizer to models/

Run: python src/train.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from preprocessing import load_and_prepare

DATA_PATH = "data/enron_spam_data.csv"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def build_param_grids():
    """Small-but-real grids. Kept tight so the whole benchmark finishes in a
    few minutes on a laptop; widen them if you have time/compute to spare."""
    return {
        "naive_bayes": {
            "estimator": MultinomialNB(),
            "params": {"alpha": [0.1, 0.5, 1.0]},
        },
        "svm": {
            # LinearSVC scales to tens of thousands of rows; kernel SVC does not.
            # Wrapped in CalibratedClassifierCV so we get predict_proba for ROC-AUC.
            "estimator": CalibratedClassifierCV(
                LinearSVC(random_state=RANDOM_STATE, max_iter=5000), cv=3
            ),
            "params": {"estimator__C": [0.1, 1, 10]},
        },
        "xgboost": {
            "estimator": XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=1,
            ),
            "params": {
                "n_estimators": [150, 300],
                "max_depth": [5],
                "learning_rate": [0.2],
            },
        },
    }


def main():
    cache_path = Path("data/clean_cache.parquet")
    print("Loading and cleaning data...")
    t0 = time.time()
    if cache_path.exists():
        import pandas as pd
        df = pd.read_parquet(cache_path)
    else:
        df = load_and_prepare(DATA_PATH)
        df.to_parquet(cache_path)
    print(f"  {len(df):,} emails ready ({df['label'].mean():.1%} spam) "
          f"[{time.time() - t0:.1f}s]")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"],
    )

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=8000, ngram_range=(1, 2), min_df=3, max_df=0.9,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    results = {}
    fitted_models = {}
    grids = build_param_grids()

    for name, cfg in grids.items():
        print(f"\nGridSearchCV: {name} ...")
        t0 = time.time()
        search = GridSearchCV(
            cfg["estimator"], cfg["params"],
            scoring="f1", cv=5, n_jobs=-1, verbose=0,
        )
        search.fit(X_train_vec, y_train)
        elapsed = time.time() - t0

        best_model = search.best_estimator_
        preds = best_model.predict(X_test_vec)
        probs = (
            best_model.predict_proba(X_test_vec)[:, 1]
            if hasattr(best_model, "predict_proba")
            else preds
        )

        metrics = {
            "best_params": search.best_params_,
            "cv_f1": search.best_score_,
            "test_accuracy": accuracy_score(y_test, preds),
            "test_f1": f1_score(y_test, preds),
            "test_roc_auc": roc_auc_score(y_test, probs),
            "train_time_s": round(elapsed, 1),
        }
        results[name] = metrics
        fitted_models[name] = best_model

        print(f"  best params: {search.best_params_}")
        print(f"  test accuracy={metrics['test_accuracy']:.4f}  "
              f"f1={metrics['test_f1']:.4f}  roc_auc={metrics['test_roc_auc']:.4f}  "
              f"[{elapsed:.1f}s]")

    best_name = max(results, key=lambda k: results[k]["test_f1"])
    best_model = fitted_models[best_name]
    preds = best_model.predict(X_test_vec)

    print(f"\n=== Best model: {best_name} ===")
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))
    cm = confusion_matrix(y_test, preds)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    false_positive_rate = fp / (fp + tn)
    print(f"False positive rate: {false_positive_rate:.4%}")

    joblib.dump(best_model, MODELS_DIR / "spam_classifier.joblib")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")

    summary = {
        "best_model": best_name,
        "false_positive_rate": false_positive_rate,
        "results": results,
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved model + vectorizer to {MODELS_DIR}/")
    print(f"Metrics written to {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
