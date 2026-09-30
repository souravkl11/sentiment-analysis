# Sentix ML: Social Media Sentiment Analysis (Classical Machine Learning)

**Academic Level:** 5th Semester B.Tech - Machine Learning Paper  
**Domain:** Natural Language Processing (NLP) & Social Media Sentiment Classification  
**Target Algorithms:** Multinomial Naive Bayes, K-Nearest Neighbors (KNN), Logistic Regression, Linear Support Vector Machine (Linear SVM)  
**Dataset:** Twitter US Airline Sentiment Dataset (14,640 annotated tweets)

---

## 1. Project Overview & Architecture

This project constructs, benchmarks, and provides an interactive web application for classical Machine Learning classification of social media text into **Positive**, **Neutral**, and **Negative** sentiment polarities.

Unlike resource-heavy deep learning architectures, this system leverages foundational NLP text vectorization (**TF-IDF unigrams + bigrams**) and classical classifiers studied in undergraduate Machine Learning coursework, delivering sub-15ms inference latency on standard CPU hardware.

### Project Structure:
```
sentiment-analysis/
├── data/
│   └── tweets.csv                      # Twitter Airline Sentiment dataset (14.6K rows)
├── src/
│   ├── __init__.py                     # Package initialization
│   ├── dataset_loader.py               # Dataset acquisition, caching & normalization
│   ├── preprocessing.py                # Regex noise filter, lowercasing & lemmatization
│   ├── feature_extraction.py           # TF-IDF Vectorizer with n-grams & feature inspection
│   ├── models.py                       # Multinomial NB, KNN, Logistic Regression, Linear SVM
│   └── visualization.py                # Confusion matrices & model comparison plots
├── static/
│   ├── index.html                      # Glassmorphic high-tech dark interface
│   ├── style.css                       # Modern CSS design tokens, animations & responsive layout
│   └── app.js                          # Client-side state, API calls & interactive visualizations
├── artifacts/
│   ├── model_suite.joblib              # Serialized trained model suite
│   ├── tfidf_vectorizer.joblib         # Fitted TF-IDF vectorizer (X_train strictly)
│   ├── metrics.json                    # Benchmark evaluation metrics & reports
│   ├── confusion_matrices.png          # 2x2 Seaborn confusion matrix heatmaps
│   └── model_comparison.png            # Matplotlib Macro F1 & Accuracy comparison plot
├── notebooks/
│   └── sentiment_analysis_demo.ipynb   # Interactive Jupyter notebook walkthrough
├── sentiment_analysis_starter.py       # End-to-end training & evaluation starter script
├── app.py                              # FastAPI web server and REST API
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```

---

## 2. Benchmark Scorecard

Evaluated on a **stratified 20% holdout test set** (2,923 tweets) with `zero_division=0`:

| Rank | Algorithm | Key Hyperparameters | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **Linear SVM** | $C=1.0$, Calibrated Probabilities | **76.02%** | **71.28%** | **65.35%** | **0.6765** |
| 🥈 | **Logistic Regression** | $C=1.0$, L2 Regularization, `max_iter=1000` | **76.02%** | 71.92% | 64.36% | **0.6715** |
| 🥉 | **Multinomial Naive Bayes** | $\alpha=1.0$ (Laplace Smoothing) | **74.03%** | 74.24% | 57.11% | **0.6101** |
| 4 | **K-Nearest Neighbors** | $k=5$, Metric: Cosine Distance | **67.36%** | 60.15% | 56.92% | **0.5815** |

### Key Theoretical Findings:
- **Linear Separability:** Linear SVM and Logistic Regression consistently outperform KNN in high-dimensional sparse text vectors ($V = 2500$ terms) due to the convex margin maximization.
- **Metric Selection:** Cosine metric for KNN effectively prevents document length bias, outperforming standard Euclidean distance in text spaces.
- **Smoothing:** Laplace smoothing in Multinomial Naive Bayes avoids zero-probability collapse on rare Twitter vocabulary.

---

## 3. Interactive Web Application

The project features a web app designed with modern glassmorphism, responsive dark theme, and interactive analytical widgets:

1. **Live Sentiment Analyzer:**
   - Real-time consensus meter (percentage agreement among all 4 models).
   - Side-by-side individual model prediction cards with 3-class probability distribution bars.
   - Active TF-IDF salient terms chips showing which n-grams drove the classification.
   - Quick preset scenarios (Positive Flight, Delay/Lost Baggage, Neutral Query, Sarcastic/Mixed).
2. **Educational NLP Pipeline Inspector:**
   - Visual stepper displaying raw text $\to$ regex cleaned text $\to$ stopword stripped & lemmatized tokens $\to$ TF-IDF numerical vector.
3. **Model Benchmarks Scorecard:**
   - Interactive Confusion Matrix heatmap viewer with per-model toggle.
   - Download links for high-res generated plots (`confusion_matrices.png`, `model_comparison.png`).
4. **Batch Analysis Studio:**
   - Bulk tweet classification with instant polarity distribution graphs (Positive %, Neutral %, Negative %).
5. **Theory Reference:**
   - Mathematical formulations, formulas, and hyperparameter justifications mapped to 5th semester ML curriculum.

---

## 4. Quickstart Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.12)

### Installation
1. Clone or navigate to the repository:
   ```bash
   cd d:\Projects\sentiment-analysis
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the End-to-End Training Pipeline
Execute the starter script to download the dataset, train all 4 classifiers, generate plots, and persist artifacts:
```bash
python sentiment_analysis_starter.py
```

### Launching the Web Application
Start the FastAPI server:
```bash
python app.py
```
Or with Uvicorn:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser and navigate to:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 5. REST API Reference

The FastAPI backend provides structured endpoints:

- `POST /api/predict`: Classify a single social media post.
  ```json
  // Request
  { "text": "@VirginAmerica amazing flight to SFO! Friendly crew, loved the mood lighting." }
  
  // Response
  {
    "cleaned_text": "flight amazing sfo friendly crew loved mood lighting",
    "top_features": [
      { "term": "amazing flight", "tfidf_weight": 0.4521 },
      { "term": "friendly crew", "tfidf_weight": 0.3842 }
    ],
    "consensus": {
      "sentiment": "positive",
      "agreement_percentage": 100.0,
      "unanimous": true
    },
    "models": { ... }
  }
  ```
- `POST /api/batch`: Classify a list of tweets and return aggregate polarity breakdown.
- `GET /api/metrics`: Retrieve academic benchmark scores and confusion matrix data.
- `GET /api/health`: Verify model readiness and loaded components.

---

## 6. Verification Checklist Compliance

- [x] **Stratified Split:** `stratify=y` strictly used in `train_test_split(X, y, test_size=0.2)` to preserve class distribution.
- [x] **No Data Leakage:** `TfidfVectorizer` fitted strictly on `X_train` and applied via `transform` to `X_test` and inference queries.
- [x] **Zero-Division Handling:** Configured `zero_division=0` in all precision, recall, and classification report computations.
- [x] **Automatic Plot Generation:** `confusion_matrices.png` and `model_comparison.png` automatically saved to `artifacts/`.
- [x] **Web Application:** Fully responsive, modern dark-mode interface with live inference and pipeline exploration.
