"""
Sentiment Analysis Starter Script
End-to-end Classical ML Pipeline for Social Media Sentiment Analysis
Follows Academic Specifications & Verification Checklist
"""

import os
import sys
import time
from sklearn.model_selection import train_test_split
import pandas as pd

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset_loader import load_data, download_dataset_if_missing
from src.preprocessing import clean_text
from src.feature_extraction import TextFeatureExtractor
from src.models import ModelSuite, get_default_classifiers
from src.visualization import evaluate_models


def run_pipeline(
    data_path: str = None,
    artifacts_dir: str = "artifacts",
    max_features: int = 2500,
    test_size: float = 0.2,
    random_state: int = 42
):
    print("=" * 70)
    print("  SOCIAL MEDIA SENTIMENT ANALYSIS PIPELINE (CLASSICAL ML)")
    print("=" * 70)

    # 1. Dataset Loading
    print("\n[Phase 1] Loading Dataset...")
    df = load_data(data_path)
    print(f"Dataset Loaded: {len(df):,} samples.")
    print("Class Distribution:")
    for sentiment, count in df["sentiment"].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"  - {sentiment.capitalize():<10}: {count:,} ({pct:.1f}%)")

    # 2. Text Preprocessing
    print("\n[Phase 2] Preprocessing Text (Lowercasing, Regex Noise Filter, Lemmatization)...")
    start_time = time.time()
    df["cleaned_text"] = df["text"].apply(clean_text)
    # Remove any empty texts after cleaning
    df = df[df["cleaned_text"].str.strip().str.len() > 0].reset_index(drop=True)
    print(f"Cleaned {len(df):,} texts in {time.time() - start_time:.2f} seconds.")

    # 3. Train-Test Split (with stratify=y to preserve class balance)
    print(f"\n[Phase 3] Splitting Train/Test (test_size={test_size}, stratify=y)...")
    X = df["cleaned_text"].values
    y = df["sentiment"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # CRITICAL: Verification checklist item 1
    )
    print(f"Training set: {len(X_train):,} samples")
    print(f"Testing set:  {len(X_test):,} samples")

    # 4. Feature Extraction (TF-IDF fit ONLY on X_train to prevent data leakage)
    print(f"\n[Phase 4] TF-IDF Feature Extraction (max_features={max_features}, ngrams=(1, 2))...")
    # CRITICAL: Verification checklist item 2 - fit only on X_train
    feature_extractor = TextFeatureExtractor(
        ngram_range=(1, 2),
        max_features=max_features,
        stop_words="english"
    )
    X_train_tfidf = feature_extractor.fit_transform(X_train)
    X_test_tfidf = feature_extractor.transform(X_test)
    print(f"TF-IDF Matrix Shape (Train): {X_train_tfidf.shape}")
    print(f"TF-IDF Matrix Shape (Test):  {X_test_tfidf.shape}")

    # 5. Model Suite Training
    print("\n[Phase 5] Training Classical ML Classifiers Suite...")
    classifiers = get_default_classifiers()
    suite = ModelSuite(classifiers=classifiers)

    for name, clf in suite.classifiers.items():
        t0 = time.time()
        print(f"  Training {name}...", end="", flush=True)
        clf.fit(X_train_tfidf, y_train)
        print(f" Done ({time.time() - t0:.2f}s)")
    suite._is_trained = True

    # 6. Evaluation & Plot Generation
    print(f"\n[Phase 6] Evaluating Models & Generating Plots in '{artifacts_dir}'...")
    # CRITICAL: Verification checklist items 3 & 4
    metrics = evaluate_models(
        suite.classifiers,
        X_test_tfidf,
        y_test,
        class_labels=["negative", "neutral", "positive"],
        save_dir=artifacts_dir
    )

    print("\n" + "-" * 70)
    print(f"{'Model Name':<28} | {'Accuracy':<10} | {'Macro Prec':<10} | {'Macro Rec':<10} | {'Macro F1':<10}")
    print("-" * 70)
    for model_name, m in metrics.items():
        print(f"{model_name:<28} | {m['accuracy']:<10.4f} | {m['precision_macro']:<10.4f} | {m['recall_macro']:<10.4f} | {m['f1_macro']:<10.4f}")
    print("-" * 70)

    # 7. Model Persistence
    print(f"\n[Phase 7] Persisting Models & Preprocessor to '{artifacts_dir}'...")
    os.makedirs(artifacts_dir, exist_ok=True)
    suite.save(artifacts_dir)
    feature_extractor.save(os.path.join(artifacts_dir, "tfidf_vectorizer.joblib"))
    print("Model artifacts successfully saved.")

    # 8. Interactive Demo Inference
    print("\n[Phase 8] Sample Inference Demonstration:")
    sample_queries = [
        "Flight was delayed by 6 hours, gate staff were unhelpful, lost my luggage! Terrible service.",
        "Had a wonderful experience on flight 302! Beautiful flight attendants, free WiFi, very comfortable.",
        "Can you please confirm what time the check-in opens for the flight to Chicago tomorrow?"
    ]

    for query in sample_queries:
        cleaned_query = clean_text(query)
        vec = feature_extractor.transform([cleaned_query])
        prediction = suite.predict_single(vec)
        consensus = prediction["consensus"]
        print(f"\nTweet: \"{query}\"")
        print(f"Cleaned: \"{cleaned_query}\"")
        print(f"Consensus Prediction: {consensus['sentiment'].upper()} ({consensus['agreement_percentage']}% agreement)")
        for m_name, res in prediction["models"].items():
            print(f"  - {m_name:<26}: {res['sentiment'].upper()} ({res['confidence']}%)")

    print("\n" + "=" * 70)
    print("  Pipeline execution completed successfully!")
    print("=" * 70)
    return suite, feature_extractor, metrics


if __name__ == "__main__":
    run_pipeline()
