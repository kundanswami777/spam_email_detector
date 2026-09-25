"""
preprocessing.py

Cleans raw Enron email text so it's ready to be turned into TF-IDF vectors.
Steps: lowercase -> strip punctuation/digits -> tokenize -> drop stopwords
(and very short tokens) -> stem.
"""

import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Make sure the two small NLTK corpora we need are present. This is a no-op
# if they're already downloaded.
for resource in ("stopwords", "punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

# Enron emails are full of forwarding headers / boilerplate that add noise
# without adding signal. Strip the most common ones before tokenizing.
BOILERPLATE_PATTERNS = [
    r"-{2,}\s*original message\s*-{2,}",
    r"-{2,}\s*forwarded by.*?-{2,}",
    r"subject\s*:",
    r"to\s*:",
    r"from\s*:",
    r"cc\s*:",
]


def clean_text(text: str) -> str:
    """Lowercase, strip boilerplate/punctuation/digits, return raw cleaned string."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text)

    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"\S+@\S+", " ", text)                    # email addresses
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)                        # digits
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_stem(text: str) -> str:
    """Tokenize, drop stopwords/short tokens, stem, and rejoin into a string
    (TfidfVectorizer wants strings, not token lists)."""
    tokens = word_tokenize(text)
    tokens = [
        STEMMER.stem(tok)
        for tok in tokens
        if tok not in STOPWORDS and len(tok) > 2
    ]
    return " ".join(tokens)


def preprocess(text: str) -> str:
    """Full pipeline: raw string -> model-ready string."""
    return tokenize_and_stem(clean_text(text))


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """Load the Enron CSV, combine subject + body, clean everything, and
    return a DataFrame with `text`, `clean_text`, and `label` (1=spam, 0=ham).
    """
    df = pd.read_csv(csv_path)

    df["Subject"] = df["Subject"].fillna("")
    df["Message"] = df["Message"].fillna("")
    df["text"] = (df["Subject"] + " " + df["Message"]).str.strip()

    df["label"] = (df["Spam/Ham"].str.lower() == "spam").astype(int)

    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    df["clean_text"] = df["text"].apply(preprocess)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    return df[["text", "clean_text", "label"]]


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/enron_spam_data.csv"
    data = load_and_prepare(path)
    print(data["label"].value_counts())
    print(data.head())
