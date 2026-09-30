"""
Web Application Backend (FastAPI)
Serves the Social Media Sentiment Analysis Interactive Application and REST API.
"""

import os
import sys
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import clean_text, inspect_pipeline_steps
from src.feature_extraction import TextFeatureExtractor
from src.models import ModelSuite, get_default_classifiers
from src.dataset_loader import load_data

app = FastAPI(
    title="Social Media Sentiment Analysis API",
    description="Classical ML Sentiment Classification Pipeline (Multinomial NB, KNN, Logistic Regression, Linear SVM)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount artifacts folder to serve generated plots (confusion matrices, comparison charts)
app.mount("/artifacts", StaticFiles(directory=ARTIFACTS_DIR), name="artifacts")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global in-memory model instances
model_suite: Optional[ModelSuite] = None
feature_extractor: Optional[TextFeatureExtractor] = None
benchmark_metrics: Optional[Dict[str, Any]] = None


def init_or_load_models():
    """Loads existing models or runs a fast train session."""
    global model_suite, feature_extractor, benchmark_metrics

    model_path = os.path.join(ARTIFACTS_DIR, "model_suite.joblib")
    tfidf_path = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib")
    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")

    if os.path.exists(model_path) and os.path.exists(tfidf_path):
        try:
            model_suite = ModelSuite.load(ARTIFACTS_DIR)
            feature_extractor = TextFeatureExtractor.load(tfidf_path)
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    benchmark_metrics = json.load(f)
            print("Successfully loaded pre-trained models and feature extractor.")
            return
        except Exception as e:
            print(f"Error loading models from disk: {e}. Re-training starter pipeline...")

    # Otherwise run pipeline
    from sentiment_analysis_starter import run_pipeline
    model_suite, feature_extractor, benchmark_metrics = run_pipeline(artifacts_dir=ARTIFACTS_DIR)


@app.on_event("startup")
def startup_event():
    init_or_load_models()


# Pydantic Schemas
class SinglePredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, example="Our flight was delayed 5 hours, no food vouchers. Horrible!")

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=200)

class PreprocessRequest(BaseModel):
    text: str = Field(...)


@app.get("/")
def serve_index():
    """Serves the main application page."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Sentiment Analysis API is active. Static frontend file pending."}


@app.get("/api/health")
def health_check():
    """Health status and model readiness."""
    return {
        "status": "healthy",
        "models_ready": model_suite is not None and model_suite._is_trained,
        "feature_extractor_ready": feature_extractor is not None and feature_extractor._is_fitted,
        "available_models": list(model_suite.classifiers.keys()) if model_suite else []
    }


@app.get("/api/metrics")
def get_metrics():
    """Returns academic benchmark evaluation metrics and plot paths."""
    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if benchmark_metrics:
        return benchmark_metrics
    raise HTTPException(status_code=404, detail="Benchmark metrics not yet computed. Run pipeline first.")


@app.post("/api/preprocess")
def preprocess_debug(req: PreprocessRequest):
    """Provides intermediate step inspection for educational breakdown."""
    return inspect_pipeline_steps(req.text)


@app.post("/api/predict")
def predict_sentiment(req: SinglePredictRequest):
    """
    Main prediction endpoint:
    - Preprocesses social media text
    - Extracts TF-IDF top terms
    - Predicts across all 4 models (Multinomial NB, KNN, Logistic Regression, Linear SVM)
    - Computes consensus agreement and probability breakdowns
    """
    if not model_suite or not feature_extractor:
        raise HTTPException(status_code=503, detail="Models are initializing. Please retry shortly.")

    pipeline_info = inspect_pipeline_steps(req.text)
    cleaned = pipeline_info["final_cleaned_text"]

    if not cleaned.strip():
        # Fallback if raw text contained only stop words or stripped punctuation
        cleaned = req.text.lower().strip()

    # TF-IDF vectorization
    vec = feature_extractor.transform([cleaned])
    prediction_result = model_suite.predict_single(vec)
    top_features = feature_extractor.get_top_features_for_text(cleaned, top_k=6)

    return {
        "original_text": req.text,
        "cleaned_text": cleaned,
        "pipeline_steps": pipeline_info,
        "top_features": top_features,
        "models": prediction_result["models"],
        "consensus": prediction_result["consensus"]
    }


@app.post("/api/batch")
def batch_predict(req: BatchPredictRequest):
    """Classifies a batch of social media posts and returns aggregate distributions."""
    if not model_suite or not feature_extractor:
        raise HTTPException(status_code=503, detail="Models are initializing. Please retry shortly.")

    results = []
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}

    for text in req.texts:
        cleaned = clean_text(text)
        if not cleaned:
            cleaned = text.lower().strip()
        vec = feature_extractor.transform([cleaned])
        pred = model_suite.predict_single(vec)
        consensus_sentiment = pred["consensus"]["sentiment"]
        sentiment_counts[consensus_sentiment] = sentiment_counts.get(consensus_sentiment, 0) + 1

        results.append({
            "text": text,
            "cleaned": cleaned,
            "sentiment": consensus_sentiment,
            "agreement": pred["consensus"]["agreement_percentage"],
            "models": {m: res["sentiment"] for m, res in pred["models"].items()}
        })

    total = len(results)
    distribution = {
        k: {
            "count": v,
            "percentage": round((v / total) * 100, 1) if total > 0 else 0
        }
        for k, v in sentiment_counts.items()
    }

    return {
        "total_analyzed": total,
        "distribution": distribution,
        "results": results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
