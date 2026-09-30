"""
Feature Extraction Module for Social Media Sentiment Analysis
Implements Phase B: TF-IDF Vectorization with n-gram support and feature inspection.
"""

from typing import Tuple, List, Dict, Optional, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


class TextFeatureExtractor:
    """
    Manages TF-IDF feature extraction ensuring no data leakage:
    - Fits strictly on training split
    - Transforms test split and inference queries
    - Offers inspection of top active features / n-grams
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 2500,
        stop_words: Optional[str] = "english",
        sublinear_tf: bool = True
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.stop_words = stop_words
        self.sublinear_tf = sublinear_tf
        self.vectorizer: Optional[TfidfVectorizer] = None
        self._is_fitted: bool = False

    def fit(self, texts: List[str]) -> "TextFeatureExtractor":
        """Fit vectorizer strictly on training data."""
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            stop_words=self.stop_words,
            sublinear_tf=self.sublinear_tf
        )
        self.vectorizer.fit(texts)
        self._is_fitted = True
        return self

    def transform(self, texts: List[str]):
        """Transform texts into TF-IDF sparse matrix."""
        if not self._is_fitted or self.vectorizer is None:
            raise ValueError("Feature extractor must be fitted before transforming.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]):
        """Fit and transform simultaneously on training split."""
        self.fit(texts)
        return self.transform(texts)

    def get_top_features_for_text(self, text: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Extracts the highest-weighted TF-IDF terms for a given input text.
        Essential for explaining model predictions in the UI.
        """
        if not self._is_fitted or self.vectorizer is None:
            return []
        
        vec = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        
        coo = vec.tocoo()
        scored_terms = sorted(zip(coo.col, coo.data), key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scored_terms[:top_k]:
            results.append({
                "term": feature_names[idx],
                "tfidf_weight": round(float(score), 4)
            })
        return results

    def save(self, filepath: str):
        """Serialize fitted vectorizer."""
        joblib.dump({
            "vectorizer": self.vectorizer,
            "ngram_range": self.ngram_range,
            "max_features": self.max_features,
            "stop_words": self.stop_words
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "TextFeatureExtractor":
        """Deserialize vectorizer from disk."""
        data = joblib.load(filepath)
        extractor = cls(
            ngram_range=data.get("ngram_range", (1, 2)),
            max_features=data.get("max_features", 2500),
            stop_words=data.get("stop_words", "english")
        )
        extractor.vectorizer = data["vectorizer"]
        extractor._is_fitted = True
        return extractor
