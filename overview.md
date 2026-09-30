# Project Brief: Social Media Sentiment Analysis (Classical Machine Learning)

**Academic Level:** 5th Semester B.Tech - Machine Learning Paper  
**Domain:** Natural Language Processing (NLP) & Social Media Sentiment Classification  
**Target Algorithms:** Multinomial Naive Bayes, K-Nearest Neighbors (KNN), Logistic Regression, Support Vector Machine (Linear SVM)  

---

## 1. Project Overview & Objectives

The goal of this project is to construct, evaluate, and benchmark a classical Machine Learning pipeline to classify social media text (tweets/microblogs) into **Positive**, **Neutral**, and **Negative** sentiment polarities.

Unlike modern deep learning models that require heavy GPU hardware, this project focuses on **foundational text vectorization (TF-IDF)** and **classical classification algorithms** studied in undergraduate Machine Learning coursework [17, 24].

### Key Learning Objectives:
- Implement end-to-end NLP data processing (regex noise removal, tokenization, stop-word filtering).
- Convert unstructured text into structured numerical matrices using **TF-IDF Vectorization**.
- Train and evaluate classical classifiers: **Multinomial Naive Bayes**, **KNN ($k=3, 5, 7$)**, **Logistic Regression**, and **Linear SVM** [7, 24].
- Compare performance across Accuracy, Precision, Recall, Macro F1-Score, and Confusion Matrix heatmaps.

---

## 2. Theoretical Context & Literature Baselines

Based on uploaded research literature:
1. **Rule-Based Baselines (VADER):** Rule-based systems use curated lexicons and syntactical rules to evaluate microblog text, emojis, and punctuation without model training [18, 24].
2. **Classical Machine Learning Baselines:** Algorithms like SVM and Naive Bayes served as early benchmark models on microblog classification datasets like TweetEval and SemEval [7, 27].
3. **Domain Challenges:** Social media text contains irregular slang, user tags (`@user`), links (`http://...`), and mixed polarity, making robust text cleaning essential before feature extraction [20, 21].

---

## 3. Recommended Datasets

1. **Twitter US Airline Sentiment Dataset** *(Primary Recommendation)*
   - **Size:** ~14,600 annotated tweets.
   - **Classes:** Positive, Neutral, Negative.
   - **Source:** Kaggle (`Tweets.csv`).
   - **Why:** Clean, well-balanced real-world dataset manageable on CPU/laptops.

2. **TweetEval Benchmark (SemEval-2017 Split)**
   - **Size:** ~10,000–20,000 tweets [5, 15].
   - **Format:** Fixed train/val/test splits [5, 27].

---

## 4. Technical Pipeline Architecture

### Phase A: Text Preprocessing (`preprocessing.py`)
- **Lowercasing:** Standardize all character casings.
- **Regex Noise Filter:** Strip URLs (`https?://\S+|www\.\S+`), user tags (`@\w+`), punctuation (`[^\w\s]`), and numbers.
- **Stop-Word Removal:** Remove uninformative English stop-words (`nltk.corpus.stopwords`).
- **Lemmatization / Stemming:** Reduce words to base forms using `WordNetLemmatizer`.

### Phase B: Feature Extraction (`features.py`)
- **TF-IDF Vectorizer (`sklearn.feature_extraction.text.TfidfVectorizer`):**
  - `ngram_range=(1, 2)` (Unigrams + Bigrams).
  - `max_features=2500` (Limit feature matrix dimensionality).
  - `stop_words='english'`.

### Phase C: Model Suite (`models.py`)
1. **Multinomial Naive Bayes:** `MultinomialNB(alpha=1.0)`
2. **K-Nearest Neighbors:** `KNeighborsClassifier(n_neighbors=5, metric='cosine')`
3. **Logistic Regression:** `LogisticRegression(max_iter=1000, C=1.0)`
4. **Support Vector Machine:** `SVC(kernel='linear', C=1.0)`

### Phase D: Evaluation & Visualization (`evaluate.py`)
- **Metrics:** Accuracy, Macro Precision, Macro Recall, Macro F1-Score.
- **Plots:**
  - Confusion Matrix Heatmaps (`seaborn.heatmap`).
  - Bar Chart comparing Macro F1-Score across all models (`matplotlib`).

---

## 5. Coding Agent Instructions & Directory Structure

When generating or extending code for this project, ensure the following modular structure:

```
social_media_sentiment/
├── data/
│   └── tweets.csv
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── models.py
│   └── visualization.py
├── notebooks/
│   └── sentiment_analysis_demo.ipynb
├── sentiment_analysis_starter.py
├── requirements.txt
└── README.md
```

### Dependencies (`requirements.txt`):
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
nltk>=3.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 6. Verification Checklist for Coding Agent

- [ ] Ensure `stratify=y` is used in `train_test_split(X, y, test_size=0.2)` to maintain class distribution.
- [ ] Fit `TfidfVectorizer` **only** on `X_train` and transform `X_test` to prevent data leakage.
- [ ] Handle division-by-zero warnings in precision/recall calculations (`zero_division=0`).
- [ ] Save output plots (`confusion_matrices.png`, `model_comparison.png`) to disk automatically.
