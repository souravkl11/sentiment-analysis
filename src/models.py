"""
Model Suite for Social Media Sentiment Analysis
Implements Phase C: Multinomial Naive Bayes, KNN, Logistic Regression, and Linear SVM.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import joblib


TARGET_CLASSES = ["negative", "neutral", "positive"]


def get_default_classifiers() -> Dict[str, Any]:
    """
    Initializes the four benchmark classical ML algorithms specified in the brief:
    1. Multinomial Naive Bayes (alpha=1.0)
    2. KNN (n_neighbors=5, metric='cosine')
    3. Logistic Regression (C=1.0, max_iter=1000)
    4. Linear SVM (kernel='linear', C=1.0 with calibrated probabilities)
    """
    return {
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(C=1.0, dual="auto", random_state=42))
    }


class ModelSuite:
    """
    Manages training, inference, consensus voting, and persistence for all four classifiers.
    """

    def __init__(self, classifiers: Optional[Dict[str, Any]] = None):
        self.classifiers = classifiers or get_default_classifiers()
        self.classes_: List[str] = TARGET_CLASSES
        self._is_trained = False

    def train_all(self, X_train, y_train) -> Dict[str, Any]:
        """Trains all 4 models sequentially."""
        trained_status = {}
        for name, clf in self.classifiers.items():
            clf.fit(X_train, y_train)
            if hasattr(clf, "classes_"):
                self.classes_ = list(clf.classes_)
            trained_status[name] = True
        self._is_trained = True
        return trained_status

    def predict_single(self, tfidf_vector) -> Dict[str, Any]:
        """
        Runs inference on a single TF-IDF vector across all four models.
        Returns prediction, confidence, probability breakdown for each, plus consensus.
        """
        predictions = {}
        votes: Dict[str, int] = {c: 0 for c in self.classes_}

        for name, clf in self.classifiers.items():
            pred = clf.predict(tfidf_vector)[0]
            votes[pred] = votes.get(pred, 0) + 1

            # Extract class probabilities
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(tfidf_vector)[0]
                prob_dict = {
                    cls_name: round(float(prob), 4)
                    for cls_name, prob in zip(clf.classes_, probs)
                }
                confidence = float(np.max(probs))
            elif hasattr(clf, "decision_function"):
                # Fallback if probability is disabled
                df = clf.decision_function(tfidf_vector)[0]
                exp_df = np.exp(df - np.max(df))
                probs = exp_df / np.sum(exp_df)
                prob_dict = {
                    cls_name: round(float(prob), 4)
                    for cls_name, prob in zip(clf.classes_, probs)
                }
                confidence = float(np.max(probs))
            else:
                prob_dict = {cls_name: (1.0 if cls_name == pred else 0.0) for cls_name in self.classes_}
                confidence = 1.0

            predictions[name] = {
                "sentiment": str(pred),
                "confidence": round(confidence * 100, 1),
                "probabilities": prob_dict
            }

        # Calculate consensus / majority vote
        consensus_sentiment = max(votes, key=votes.get)
        agreement_count = votes[consensus_sentiment]
        agreement_percentage = round((agreement_count / len(self.classifiers)) * 100, 1)

        return {
            "models": predictions,
            "consensus": {
                "sentiment": consensus_sentiment,
                "agreement_percentage": agreement_percentage,
                "votes": votes,
                "unanimous": (agreement_count == len(self.classifiers))
            }
        }

    def save(self, directory: str):
        """Save all model weights and metadata to a directory."""
        import os
        os.makedirs(directory, exist_ok=True)
        joblib.dump({
            "classifiers": self.classifiers,
            "classes_": self.classes_,
            "_is_trained": self._is_trained
        }, os.path.join(directory, "model_suite.joblib"))

    @classmethod
    def load(cls, directory: str) -> "ModelSuite":
        """Load trained suite from disk."""
        import os
        filepath = os.path.join(directory, "model_suite.joblib")
        data = joblib.load(filepath)
        suite = cls(classifiers=data["classifiers"])
        suite.classes_ = data.get("classes_", TARGET_CLASSES)
        suite._is_trained = data.get("_is_trained", True)
        return suite
