"""
Text Preprocessing Module for Social Media Sentiment Analysis
Implements Phase A: Lowercasing, Regex noise filtering, Stop-word removal, and Lemmatization.
"""

import re
import string
from typing import List, Dict, Any, Union
import nltk

# Ensure necessary NLTK data can be downloaded silently if needed
_NLTK_PACKAGES = ["stopwords", "wordnet", "omw-1.4", "punkt"]
for pkg in _NLTK_PACKAGES:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    STOPWORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    # High-coverage fallback stopwords set in case NLTK data is unavailable offline
    STOPWORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
        "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
        "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
        "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
        "at", "by", "for", "with", "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
        "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
        "don", "should", "now"
    }
    LEMMATIZER = None


def remove_noise(text: str) -> str:
    """
    Removes social media artifacts:
    - URLs (https?://... or www....)
    - User mentions (@username)
    - HTML entities (&amp;, &lt;, etc.)
    - Numbers and digits
    - Punctuation and non-word characters
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Decode HTML entities if any
    text = re.sub(r"&[a-z]+;", " ", text)
    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # 3. Remove user handles/tags (@user)
    text = re.sub(r"@\w+", " ", text)
    # 4. Remove hashtags symbol but keep hashtag word
    text = re.sub(r"#(\w+)", r"\1", text)
    # 5. Remove numbers and digits
    text = re.sub(r"\d+", " ", text)
    # 6. Remove punctuation and special characters
    text = re.sub(r"[^\w\s]", " ", text)
    # 7. Collapse repeated whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_word(word: str) -> str:
    """Lemmatizes a single word using WordNetLemmatizer if available."""
    if LEMMATIZER:
        try:
            return LEMMATIZER.lemmatize(word)
        except Exception:
            return word
    # Fallback basic suffix strip for plural/verbs if WordNet isn't available
    if len(word) > 4 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 3 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline:
    1. Lowercase
    2. Strip URLs, mentions, digits, punctuation
    3. Tokenize on whitespace
    4. Remove stop-words and single characters
    5. Lemmatize tokens
    6. Re-join into normalized string
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercasing
    text = text.lower()
    
    # Regex noise filter
    text = remove_noise(text)
    
    # Tokenize
    tokens = text.split()
    
    # Stop-word removal and lemmatization
    cleaned_tokens = [
        lemmatize_word(token)
        for token in tokens
        if token not in STOPWORDS and len(token) > 1
    ]
    
    return " ".join(cleaned_tokens)


def inspect_pipeline_steps(text: str) -> Dict[str, Any]:
    """
    Provides granular intermediate steps for educational & visual inspection.
    Useful for the web application's NLP pipeline visualizer.
    """
    raw = str(text) if text is not None else ""
    lowercased = raw.lower()
    noise_filtered = remove_noise(lowercased)
    tokens = noise_filtered.split()
    
    tokens_no_stopwords = [t for t in tokens if t not in STOPWORDS]
    lemmatized = [lemmatize_word(t) for t in tokens_no_stopwords if len(t) > 1]
    final_text = " ".join(lemmatized)
    
    return {
        "original": raw,
        "lowercased": lowercased,
        "noise_filtered": noise_filtered,
        "initial_tokens": tokens,
        "stopwords_removed": tokens_no_stopwords,
        "lemmatized_tokens": lemmatized,
        "final_cleaned_text": final_text,
        "tokens_count_before": len(raw.split()),
        "tokens_count_after": len(lemmatized)
    }
