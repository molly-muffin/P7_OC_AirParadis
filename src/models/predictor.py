"""Load production sentiment model and run predictions."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_DIR = ROOT / "models" / "production"


class SentimentPredictor:
    """Unified predictor for sklearn, Keras, and DistilBERT bundles."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self._read_production_marker()
        self.bundle: Dict[str, Any] = {}
        self.keras_model = None
        self.bert_model = None
        self.bert_tokenizer = None
        self._load()

    def _read_production_marker(self) -> str:
        env_model = os.getenv("PRODUCTION_MODEL")
        if env_model:
            return env_model.strip()
        marker = PRODUCTION_DIR / "production_model.txt"
        if marker.exists():
            return marker.read_text().strip()
        bundles = list(PRODUCTION_DIR.glob("*_bundle.pkl"))
        if bundles:
            return bundles[0].stem.replace("_bundle", "")
        return "tfidf_logistic"

    def _load(self) -> None:
        bert_dir = PRODUCTION_DIR / "distilbert_finetuned"
        if self.model_name == "distilbert_finetuned" and bert_dir.exists():
            self._load_bert(str(bert_dir))
            return

        bundle_path = PRODUCTION_DIR / f"{self.model_name}_bundle.pkl"
        if bundle_path.exists():
            with open(bundle_path, "rb") as f:
                self.bundle = pickle.load(f)
            if self.bundle.get("model_type") == "keras_cnn_lstm":
                import tensorflow as tf

                self.keras_model = tf.keras.models.load_model(self.bundle["model_path"])
            return

        sklearn_path = PRODUCTION_DIR / "tfidf_logistic_bundle.pkl"
        if sklearn_path.exists():
            with open(sklearn_path, "rb") as f:
                self.bundle = pickle.load(f)
            self.model_name = "tfidf_logistic"
            return

        if bert_dir.exists():
            self._load_bert(str(bert_dir))
            self.model_name = "distilbert_finetuned"

    def _load_bert(self, model_dir: str) -> None:
        import torch
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

        self.bert_tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.bert_model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.bert_model.eval()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model.to(self._device)

    def predict(self, text: str) -> Tuple[str, float]:
        """Return (sentiment_label, confidence) where label is 'positive' or 'negative'."""
        if self.keras_model is not None:
            return self._predict_keras(text)
        if self.bert_model is not None:
            return self._predict_bert(text)
        return self._predict_sklearn(text)

    def _predict_sklearn(self, text: str) -> Tuple[str, float]:
        vectorizer = self.bundle["vectorizer"]
        model = self.bundle["model"]
        vec = vectorizer.transform([text])
        proba = model.predict_proba(vec)[0]
        label_idx = int(np.argmax(proba))
        confidence = float(proba[label_idx])
        label = "positive" if label_idx == 1 else "negative"
        return label, confidence

    def _predict_keras(self, text: str) -> Tuple[str, float]:
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        tokenizer = self.bundle["tokenizer"]
        max_len = self.bundle["max_len"]
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
        proba = float(self.keras_model.predict(padded, verbose=0)[0][0])
        label = "positive" if proba >= 0.5 else "negative"
        confidence = proba if label == "positive" else 1.0 - proba
        return label, confidence

    def _predict_bert(self, text: str) -> Tuple[str, float]:
        import torch

        enc = self.bert_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.bert_model(**enc).logits
            proba = torch.softmax(logits, dim=1)[0].cpu().numpy()
        label_idx = int(np.argmax(proba))
        label = "positive" if label_idx == 1 else "negative"
        return label, float(proba[label_idx])

    def info(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "bundle_keys": list(self.bundle.keys())}
