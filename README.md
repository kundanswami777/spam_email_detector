# Spam Email Detection using NLP and ML

An end-to-end pipeline that classifies emails as spam or ham, trained on the
full [Enron spam dataset](https://github.com/MWiechmann/enron_spam_data)
(33,650 usable emails after cleaning). Text goes in, a spam/ham label comes
out, with the model choice benchmarked rather than assumed.

## How it works

1. **Clean the text.** Subject + body are lowercased, stripped of forwarding
   headers/URLs/emails/punctuation/digits, tokenized, stopwords removed, and
   stemmed (`src/preprocessing.py`).
2. **Vectorize.** TF-IDF with unigrams + bigrams, capped at 8,000 features.
3. **Benchmark three classifiers** with `GridSearchCV` (5-fold, scored on
   F1): Multinomial Naive Bayes, Linear SVM, and XGBoost.
4. **Pick the winner** by test-set F1, then report accuracy, F1, ROC-AUC, and
   a confusion matrix for it.
5. **Save** the winning model and the fitted vectorizer with `joblib` so you
   can classify new emails without retraining.

## Results

Run on the full dataset (80/20 train/test split, stratified):

| Model | Best params | Test accuracy | Test F1 | Test ROC-AUC |
|---|---|---|---|---|
| Naive Bayes | `alpha=0.1` | 98.54% | 0.9857 | 0.9983 |
| **SVM (winner)** | `C=1` | **99.15%** | **0.9917** | **0.9993** |
| XGBoost | `n_estimators=300, max_depth=5, lr=0.2` | 98.72% | 0.9875 | 0.9989 |

The SVM (a `LinearSVC`, wrapped in `CalibratedClassifierCV` so it can still
output probabilities) came out on top. On the held-out test set it misses
23 spam emails as ham and flags 34 ham emails as spam, out of 6,730 —
a **false positive rate of about 1.03%**.

A couple of notes on the numbers: a kernel SVM (`SVC`) doesn't scale well
past a few thousand rows, so `LinearSVC` stands in for "SVM" here — it's the
right tool for a dataset this size and it's what actually got benchmarked.
XGBoost's grid search is also the slow part of the pipeline (the tree
ensemble just takes longer to fit than the other two); it's still fast to
*predict* with once trained.

## Project structure

```
spam_email_detector/
├── data/
│   └── enron_spam_data.csv      # not committed — see "Get the data" below
├── models/                      # trained model + vectorizer land here
├── src/
│   ├── preprocessing.py         # text cleaning + tokenizing
│   ├── train.py                 # benchmarks all 3 models, saves the best
│   └── predict.py               # classify new email text from the CLI
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone https://github.com/kundanswami777/spam_email_detector.git
cd spam_email_detector
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt punkt_tab
```

### Get the data

The raw CSV (~50MB) isn't committed to keep the repo small. Grab it from the
source dataset:

```bash
curl -L -o enron_repo.zip https://codeload.github.com/MWiechmann/enron_spam_data/zip/refs/heads/master
unzip -p enron_repo.zip enron_spam_data-master/enron_spam_data.zip > data/enron_spam_data.zip
unzip -o data/enron_spam_data.zip -d data/
```

### Train

```bash
python src/train.py
```

This cleans the data (cached to `data/clean_cache.parquet` after the first
run so reruns are fast), benchmarks all three models with 5-fold
`GridSearchCV`, prints a classification report for the best one, and saves
`models/spam_classifier.joblib` + `models/tfidf_vectorizer.joblib`.

### Classify an email

```bash
python src/predict.py "You've won a prize!" "Click here to claim your free reward now!!!"
# Prediction: SPAM  Confidence: 99.8%

python src/predict.py --file some_email.txt
```

## Why these choices

- **TF-IDF over word embeddings** — with a vocabulary this size and a linear
  model doing the heavy lifting, TF-IDF is cheap, interpretable, and
  performs essentially as well as anything fancier here.
- **Bigrams, not just unigrams** — phrases like "act now" or "free money"
  carry more spam signal together than either word alone.
- **F1 as the grid-search metric** — accuracy alone can look good on a
  roughly balanced dataset while still hiding a lopsided precision/recall
  trade-off; F1 keeps both in view.

## Possible extensions

- Swap TF-IDF for sentence embeddings and compare
- Add SHAP/LIME explanations for individual predictions
- Wrap `predict.py` in a small Flask/FastAPI endpoint
- Track experiments with MLflow instead of a flat `metrics.json`
