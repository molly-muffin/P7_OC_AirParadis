"""
End-to-end training pipeline for Air Paradis sentiment analysis.

Trains simple (TF-IDF), advanced (CNN-LSTM + embeddings), and BERT models
with MLflow tracking and saves production artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.cleaned_loader import load_cleaned_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_URI = f"file://{(ROOT / 'mlruns').absolute()}"
PRODUCTION_DIR = ROOT / "models" / "production"
RESULTS_FILE = PRODUCTION_DIR / "model_comparison.json"


def _metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _log_sklearn_run(
    run_name: str,
    model,
    vectorizer: TfidfVectorizer,
    params: Dict[str, Any],
    y_val,
    y_val_pred,
    y_test,
    y_test_pred,
) -> Dict[str, Any]:
    val_m = _metrics(y_val, y_val_pred)
    test_m = _metrics(y_test, y_test_pred)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        for k, v in val_m.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_m.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.sklearn.log_model(model, "model")
        artifact = {"vectorizer": vectorizer, "model": model, "model_type": "sklearn_tfidf"}
        path = PRODUCTION_DIR / f"{run_name}_bundle.pkl"
        PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(artifact, f)
        mlflow.log_artifact(str(path))

    return {"model": run_name, **{f"val_{k}": v for k, v in val_m.items()}, **{f"test_{k}": v for k, v in test_m.items()}}


def train_simple_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Train TF-IDF + LogisticRegression and RandomForest."""
    results = []

    X_train = train_df["cleaned_text"]
    y_train = train_df["target"]
    X_val = val_df["cleaned_text"]
    y_val = val_df["target"]
    X_test = test_df["cleaned_text"]
    y_test = test_df["target"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_vec, y_train)
    results.append(
        _log_sklearn_run(
            "tfidf_logistic",
            log_reg,
            vectorizer,
            {"max_features": 5000, "classifier": "LogisticRegression"},
            y_val,
            log_reg.predict(X_val_vec),
            y_test,
            log_reg.predict(X_test_vec),
        )
    )

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_train_vec, y_train)
    results.append(
        _log_sklearn_run(
            "tfidf_random_forest",
            rf,
            vectorizer,
            {"max_features": 5000, "n_estimators": 100, "max_depth": 10},
            y_val,
            rf.predict(X_val_vec),
            y_test,
            rf.predict(X_test_vec),
        )
    )

    return results


def _tokenize(texts: pd.Series) -> List[List[str]]:
    return [str(t).split() for t in texts]


def _build_embedding_matrix(tokenizer_word_index: Dict[str, int], embedding_model, dim: int) -> np.ndarray:
    vocab_size = len(tokenizer_word_index) + 1
    matrix = np.random.normal(scale=0.6, size=(vocab_size, dim)).astype(np.float32)
    matrix[0] = 0.0
    for word, idx in tokenizer_word_index.items():
        if word in embedding_model:
            matrix[idx] = embedding_model[word]
    return matrix


def _mean_embedding_features(texts: List[List[str]], embedding_model, dim: int) -> np.ndarray:
    vectors = []
    for tokens in texts:
        word_vecs = [embedding_model[w] for w in tokens if w in embedding_model]
        if word_vecs:
            vectors.append(np.mean(word_vecs, axis=0))
        else:
            vectors.append(np.zeros(dim))
    return np.array(vectors)


def train_embedding_classical(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    embedding_name: str,
) -> Dict[str, Any]:
    """Train LogisticRegression on mean word embeddings (Word2Vec or GloVe)."""
    from gensim.models import Word2Vec

    train_tokens = _tokenize(train_df["cleaned_text"])
    val_tokens = _tokenize(val_df["cleaned_text"])
    test_tokens = _tokenize(test_df["cleaned_text"])

    dim = 100
    if embedding_name == "word2vec":
        w2v = Word2Vec(
            sentences=train_tokens,
            vector_size=dim,
            window=5,
            min_count=2,
            workers=2,
            epochs=5,
            seed=42,
        )
        emb = w2v.wv
    else:
        try:
            import gensim.downloader as api

            logger.info("Loading GloVe twitter-25 via gensim downloader...")
            emb = api.load("glove-twitter-25")
            dim = emb.vector_size
        except Exception as exc:
            logger.warning("GloVe download failed (%s), using Word2Vec CBOW fallback", exc)
            w2v = Word2Vec(
                sentences=train_tokens,
                vector_size=dim,
                window=5,
                min_count=2,
                workers=2,
                epochs=5,
                sg=0,
                seed=42,
            )
            emb = w2v.wv
            embedding_name = "glove_fallback_cbow"

    X_train = _mean_embedding_features(train_tokens, emb, dim)
    X_val = _mean_embedding_features(val_tokens, emb, dim)
    X_test = _mean_embedding_features(test_tokens, emb, dim)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, train_df["target"])

    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)
    val_m = _metrics(val_df["target"], y_val_pred)
    test_m = _metrics(test_df["target"], y_test_pred)

    run_name = f"embedding_{embedding_name}_logreg"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"embedding": embedding_name, "dim": dim})
        for k, v in val_m.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_m.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.sklearn.log_model(clf, "model")

    return {
        "model": run_name,
        **{f"val_{k}": v for k, v in val_m.items()},
        **{f"test_{k}": v for k, v in test_m.items()},
        "embedding": embedding_name,
    }


def train_cnn_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    embedding_name: str = "word2vec",
    epochs: int = 3,
) -> Dict[str, Any]:
    """Train hybrid CNN-LSTM model with embeddings."""
    import tensorflow as tf
    from gensim.models import Word2Vec
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import (
        Conv1D,
        Dense,
        Dropout,
        Embedding,
        GlobalMaxPooling1D,
        Input,
        LSTM,
        MaxPooling1D,
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    tf.random.set_seed(42)

    max_features = 10000
    max_len = 80
    emb_dim = 100

    tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["cleaned_text"].astype(str))

    def to_sequences(texts):
        seq = tokenizer.texts_to_sequences(texts)
        return pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    X_train = to_sequences(train_df["cleaned_text"])
    X_val = to_sequences(val_df["cleaned_text"])
    X_test = to_sequences(test_df["cleaned_text"])
    y_train = train_df["target"].values
    y_val = val_df["target"].values
    y_test = test_df["target"].values

    train_tokens = _tokenize(train_df["cleaned_text"])
    if embedding_name == "word2vec":
        w2v = Word2Vec(
            sentences=train_tokens,
            vector_size=emb_dim,
            window=5,
            min_count=2,
            workers=2,
            epochs=5,
            seed=42,
        )
        emb = w2v.wv
    else:
        try:
            import gensim.downloader as api

            emb = api.load("glove-twitter-25")
            emb_dim = emb.vector_size
        except Exception:
            w2v = Word2Vec(
                sentences=train_tokens,
                vector_size=emb_dim,
                window=5,
                min_count=2,
                workers=2,
                epochs=5,
                sg=0,
                seed=42,
            )
            emb = w2v.wv
            embedding_name = "glove_fallback_cbow"

    embedding_matrix = _build_embedding_matrix(tokenizer.word_index, emb, emb_dim)

    inp = Input(shape=(max_len,))
    x = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=emb_dim,
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=False,
    )(inp)
    x = Conv1D(128, 5, activation="relu")(x)
    x = MaxPooling1D(5)(x)
    x = LSTM(64, dropout=0.3)(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    early = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=64,
        callbacks=[early],
        verbose=1,
    )

    y_val_pred = (model.predict(X_val, verbose=0).flatten() >= 0.5).astype(int)
    y_test_pred = (model.predict(X_test, verbose=0).flatten() >= 0.5).astype(int)
    val_m = _metrics(y_val, y_val_pred)
    test_m = _metrics(y_test, y_test_pred)

    run_name = f"cnn_lstm_{embedding_name}"
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    model_path = PRODUCTION_DIR / f"{run_name}.keras"
    bundle_path = PRODUCTION_DIR / f"{run_name}_bundle.pkl"
    model.save(model_path)

    bundle = {
        "model_type": "keras_cnn_lstm",
        "model_path": str(model_path),
        "tokenizer": tokenizer,
        "max_len": max_len,
        "embedding": embedding_name,
    }
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"embedding": embedding_name, "epochs": epochs, "max_len": max_len})
        for k, v in val_m.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_m.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(bundle_path))

    return {
        "model": run_name,
        **{f"val_{k}": v for k, v in val_m.items()},
        **{f"test_{k}": v for k, v in test_m.items()},
        "bundle_path": str(bundle_path),
        "val_f1": val_m["f1"],
    }


def train_distilbert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int = 2,
) -> Dict[str, Any]:
    """Fine-tune DistilBERT for sentiment classification."""
    import os

    cache = ROOT / "data" / "cache" / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache))

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizer,
        get_linear_schedule_with_warmup,
    )

    class TweetDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                str(self.texts[idx]),
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    train_loader = DataLoader(
        TweetDataset(train_df["cleaned_text"].tolist(), train_df["target"].tolist(), tokenizer),
        batch_size=16,
        shuffle=True,
    )
    val_loader = DataLoader(
        TweetDataset(val_df["cleaned_text"].tolist(), val_df["target"].tolist(), tokenizer),
        batch_size=32,
    )
    test_loader = DataLoader(
        TweetDataset(test_df["cleaned_text"].tolist(), test_df["target"].tolist(), tokenizer),
        batch_size=32,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

    def predict_loader(loader) -> np.ndarray:
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        return np.array(preds)

    y_val_pred = predict_loader(val_loader)
    y_test_pred = predict_loader(test_loader)
    val_m = _metrics(val_df["target"], y_val_pred)
    test_m = _metrics(test_df["target"], y_test_pred)

    run_name = "distilbert_finetuned"
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = PRODUCTION_DIR / run_name
    model_dir.mkdir(exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"model": model_name, "epochs": epochs})
        for k, v in val_m.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_m.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_artifact(str(model_dir))

    return {
        "model": run_name,
        **{f"val_{k}": v for k, v in val_m.items()},
        **{f"test_{k}": v for k, v in test_m.items()},
        "model_dir": str(model_dir),
        "val_f1": val_m["f1"],
    }


def select_production_model(results: List[Dict[str, Any]]) -> str:
    """Pick best model by validation F1 (prefer BERT/DL over simple if close)."""
    if not results:
        return "tfidf_logistic"

    def score(r: Dict[str, Any]) -> float:
        return r.get("val_f1", r.get("val_accuracy", 0))

    best = max(results, key=score)
    marker = PRODUCTION_DIR / "production_model.txt"
    marker.write_text(best["model"])
    logger.info("Production model selected: %s (val_f1=%.4f)", best["model"], score(best))
    return best["model"]


def run_pipeline(sample_size: Optional[int] = 50000, skip_bert: bool = False) -> List[Dict[str, Any]]:
    """Run full training pipeline."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("air-paradis-sentiment")

    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data (sample_size=%s)...", sample_size)
    train_df, val_df, test_df = load_cleaned_dataset(sample_size=sample_size)

    all_results: List[Dict[str, Any]] = []
    all_results.extend(train_simple_models(train_df, val_df, test_df))
    all_results.append(train_embedding_classical(train_df, val_df, test_df, "word2vec"))
    all_results.append(train_embedding_classical(train_df, val_df, test_df, "glove"))
    all_results.append(train_cnn_lstm(train_df, val_df, test_df, "word2vec", epochs=3))
    all_results.append(train_cnn_lstm(train_df, val_df, test_df, "glove", epochs=3))

    if not skip_bert:
        bert_sample = min(sample_size or 50000, 20000)
        b_train, b_val, b_test = load_cleaned_dataset(sample_size=bert_sample)
        all_results.append(train_distilbert(b_train, b_val, b_test, epochs=2))

    select_production_model(all_results)

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("Results saved to %s", RESULTS_FILE)
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Train Air Paradis sentiment models")
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--skip-bert", action="store_true")
    parser.add_argument("--model", choices=["all", "simple", "dl", "bert"], default="all")
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("air-paradis-sentiment")
    train_df, val_df, test_df = load_cleaned_dataset(sample_size=args.sample_size)
    results = []

    if args.model in ("all", "simple"):
        results.extend(train_simple_models(train_df, val_df, test_df))
    if args.model in ("all", "dl"):
        results.append(train_cnn_lstm(train_df, val_df, test_df, "word2vec"))
        results.append(train_cnn_lstm(train_df, val_df, test_df, "glove"))
    if args.model in ("all", "bert") and not args.skip_bert:
        b_train, b_val, b_test = load_cleaned_dataset(sample_size=min(args.sample_size, 20000))
        results.append(train_distilbert(b_train, b_val, b_test))

    if results:
        select_production_model(results)
        PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
