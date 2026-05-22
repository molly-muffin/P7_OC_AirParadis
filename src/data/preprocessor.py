"""
Text preprocessing module for sentiment analysis.

This module handles:
- Tweet cleaning (URLs, mentions, hashtags)
- Tokenization and normalization
- Emoji handling and conversion
- Feature engineering (length, hashtag count, etc.)
- Word embeddings (Word2Vec, GloVe, FastText)
"""

import re
import string
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import emoji

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Comprehensive text preprocessor for tweet sentiment analysis.
    """
    
    def __init__(self, cache_dir: str = "data/cache/"):
        """
        Initialize TextPreprocessor.
        
        Args:
            cache_dir: Directory to cache preprocessing artifacts
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize NLTK tools
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Preprocessing parameters
        self.min_word_length = 2
        self.max_features = 10000
        
        # Compiled regex patterns for efficiency
        self.patterns = self._compile_patterns()
        
        # Vectorizers
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        
        # Word embeddings
        self.word2vec_model = None
        self.fasttext_model = None
        self.glove_embeddings = None
        
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for text cleaning."""
        return {
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'mention': re.compile(r'@[A-Za-z0-9_]+'),
            'hashtag': re.compile(r'#[A-Za-z0-9_]+'),
            'rt': re.compile(r'\bRT\b', re.IGNORECASE),
            'multiple_spaces': re.compile(r'\s+'),
            'non_alphanumeric': re.compile(r'[^a-zA-Z0-9\s]'),
            'numbers': re.compile(r'\d+'),
            'repeated_chars': re.compile(r'(.)\1{2,}')
        }
    
    def clean_text(self, text: str, preserve_case: bool = False) -> str:
        """
        Clean and normalize tweet text.
        
        Args:
            text: Raw tweet text
            preserve_case: Whether to preserve original case
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert emojis to text descriptions
        text = emoji.demojize(text, delimiters=(" ", " "))
        
        # Remove URLs
        text = self.patterns['url'].sub(' ', text)
        
        # Remove mentions but keep the context
        text = self.patterns['mention'].sub(' USER ', text)
        
        # Remove hashtag symbols but keep the text
        text = self.patterns['hashtag'].sub(lambda m: m.group(0)[1:], text)
        
        # Remove RT markers
        text = self.patterns['rt'].sub('', text)
        
        # Handle repeated characters (e.g., "sooooo" -> "soo")
        text = self.patterns['repeated_chars'].sub(r'\1\1', text)
        
        # Convert to lowercase if not preserving case
        if not preserve_case:
            text = text.lower()
        
        # Remove extra whitespace
        text = self.patterns['multiple_spaces'].sub(' ', text)
        
        return text.strip()
    
    def tokenize_text(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Tokenize and filter text.
        
        Args:
            text: Cleaned text
            remove_stopwords: Whether to remove stop words
            
        Returns:
            List of tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Skip if too short
            if len(token) < self.min_word_length:
                continue
            
            # Skip if all punctuation
            if all(c in string.punctuation for c in token):
                continue
            
            # Skip stop words if requested
            if remove_stopwords and token.lower() in self.stop_words:
                continue
            
            # Remove remaining punctuation
            token = re.sub(r'[^\w]', '', token)
            
            if token:  # Only add non-empty tokens
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply stemming to tokens."""
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Apply lemmatization to tokens."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract hand-crafted features from text.
        
        Args:
            text: Original text
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Character statistics
        features['uppercase_count'] = sum(1 for c in text if c.isupper())
        features['punctuation_count'] = sum(1 for c in text if c in string.punctuation)
        features['digit_count'] = sum(1 for c in text if c.isdigit())
        
        # Twitter-specific features
        features['hashtag_count'] = len(re.findall(r'#\w+', text))
        features['mention_count'] = len(re.findall(r'@\w+', text))
        features['url_count'] = len(re.findall(r'http[s]?://\S+', text))
        features['emoji_count'] = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text))
        
        # Sentiment indicators
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = features['uppercase_count'] / len(text) if text else 0
        
        # Repeated characters/words
        features['repeated_chars'] = len(re.findall(r'(.)\1{2,}', text))
        
        return features
    
    def preprocess_dataframe(
        self, 
        df: pd.DataFrame, 
        text_column: str = 'text',
        target_column: str = 'target',
        add_features: bool = True
    ) -> pd.DataFrame:
        """
        Preprocess entire dataframe.
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            target_column: Name of target column
            add_features: Whether to add extracted features
            
        Returns:
            Preprocessed dataframe
        """
        logger.info(f"Preprocessing {len(df)} samples...")
        
        df = df.copy()
        
        # Clean text
        logger.info("Cleaning text...")
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Tokenize
        logger.info("Tokenizing text...")
        df['tokens'] = df['cleaned_text'].apply(self.tokenize_text)
        df['tokens_str'] = df['tokens'].apply(' '.join)
        
        # Extract features if requested
        if add_features:
            logger.info("Extracting features...")
            feature_dicts = df[text_column].apply(self.extract_features)
            feature_df = pd.DataFrame(feature_dicts.tolist())
            df = pd.concat([df, feature_df], axis=1)
        
        logger.info("Preprocessing completed")
        return df
    
    def fit_vectorizers(self, texts: List[str]) -> None:
        """
        Fit TF-IDF and Count vectorizers.
        
        Args:
            texts: List of preprocessed texts
        """
        logger.info("Fitting vectorizers...")
        
        # TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            strip_accents='ascii',
            lowercase=True
        )
        self.tfidf_vectorizer.fit(texts)
        
        # Count Vectorizer
        self.count_vectorizer = CountVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            strip_accents='ascii',
            lowercase=True
        )
        self.count_vectorizer.fit(texts)
        
        # Save vectorizers
        self._save_vectorizers()
        
        logger.info("Vectorizers fitted and saved")
    
    def transform_texts(self, texts: List[str], method: str = 'tfidf') -> np.ndarray:
        """
        Transform texts using fitted vectorizers.
        
        Args:
            texts: List of texts to transform
            method: 'tfidf' or 'count'
            
        Returns:
            Transformed feature matrix
        """
        if method == 'tfidf':
            if self.tfidf_vectorizer is None:
                raise ValueError("TF-IDF vectorizer not fitted. Call fit_vectorizers() first.")
            return self.tfidf_vectorizer.transform(texts)
        elif method == 'count':
            if self.count_vectorizer is None:
                raise ValueError("Count vectorizer not fitted. Call fit_vectorizers() first.")
            return self.count_vectorizer.transform(texts)
        else:
            raise ValueError("Method must be 'tfidf' or 'count'")
    
    def train_word2vec(self, tokenized_texts: List[List[str]], **kwargs) -> None:
        """
        Train Word2Vec model.
        
        Args:
            tokenized_texts: List of tokenized texts
            **kwargs: Additional arguments for Word2Vec
        """
        logger.info("Training Word2Vec model...")
        from gensim.models import Word2Vec

        default_params = {
            'vector_size': 300,
            'window': 5,
            'min_count': 2,
            'workers': 4,
            'epochs': 10,
            'sg': 1  # Skip-gram
        }
        default_params.update(kwargs)
        
        self.word2vec_model = Word2Vec(tokenized_texts, **default_params)
        
        # Save model
        model_path = self.cache_dir / 'word2vec_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.word2vec_model, f)
        
        logger.info(f"Word2Vec model trained and saved to {model_path}")
    
    def train_fasttext(self, tokenized_texts: List[List[str]], **kwargs) -> None:
        """
        Train FastText model.
        
        Args:
            tokenized_texts: List of tokenized texts
            **kwargs: Additional arguments for FastText
        """
        logger.info("Training FastText model...")
        from gensim.models import FastText

        default_params = {
            'vector_size': 300,
            'window': 5,
            'min_count': 2,
            'workers': 4,
            'epochs': 10,
            'sg': 1,  # Skip-gram
            'min_n': 3,
            'max_n': 6
        }
        default_params.update(kwargs)
        
        self.fasttext_model = FastText(tokenized_texts, **default_params)
        
        # Save model
        model_path = self.cache_dir / 'fasttext_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.fasttext_model, f)
        
        logger.info(f"FastText model trained and saved to {model_path}")
    
    def get_embeddings_matrix(
        self, 
        vocabulary: List[str], 
        embedding_type: str = 'word2vec'
    ) -> np.ndarray:
        """
        Create embeddings matrix for given vocabulary.
        
        Args:
            vocabulary: List of words
            embedding_type: 'word2vec' or 'fasttext'
            
        Returns:
            Embeddings matrix
        """
        if embedding_type == 'word2vec':
            if self.word2vec_model is None:
                raise ValueError("Word2Vec model not trained")
            model = self.word2vec_model
        elif embedding_type == 'fasttext':
            if self.fasttext_model is None:
                raise ValueError("FastText model not trained")
            model = self.fasttext_model
        else:
            raise ValueError("embedding_type must be 'word2vec' or 'fasttext'")
        
        embedding_dim = model.wv.vector_size
        embeddings_matrix = np.zeros((len(vocabulary), embedding_dim))
        
        for i, word in enumerate(vocabulary):
            if word in model.wv:
                embeddings_matrix[i] = model.wv[word]
            else:
                # Random initialization for unknown words
                embeddings_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
        
        return embeddings_matrix
    
    def _save_vectorizers(self) -> None:
        """Save fitted vectorizers to cache."""
        if self.tfidf_vectorizer is not None:
            with open(self.cache_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
        
        if self.count_vectorizer is not None:
            with open(self.cache_dir / 'count_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.count_vectorizer, f)
    
    def load_vectorizers(self) -> None:
        """Load vectorizers from cache."""
        tfidf_path = self.cache_dir / 'tfidf_vectorizer.pkl'
        count_path = self.cache_dir / 'count_vectorizer.pkl'
        
        if tfidf_path.exists():
            with open(tfidf_path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
        
        if count_path.exists():
            with open(count_path, 'rb') as f:
                self.count_vectorizer = pickle.load(f)
    
    def load_word_embeddings(self) -> None:
        """Load word embedding models from cache."""
        w2v_path = self.cache_dir / 'word2vec_model.pkl'
        ft_path = self.cache_dir / 'fasttext_model.pkl'
        
        if w2v_path.exists():
            with open(w2v_path, 'rb') as f:
                self.word2vec_model = pickle.load(f)
        
        if ft_path.exists():
            with open(ft_path, 'rb') as f:
                self.fasttext_model = pickle.load(f)


def main():
    """Example usage of TextPreprocessor."""
    # Sample tweets for testing
    sample_tweets = [
        "I love this airline! Great service 😊 #AirParadis",
        "@AirParadis your flight was delayed AGAIN!!! Not happy 😡",
        "RT @user: Check out this amazing deal http://example.com",
        "Flying with AirParadis tomorrow... hope it goes well 🤞"
    ]
    
    preprocessor = TextPreprocessor()
    
    # Test text cleaning
    print("Text Cleaning Examples:")
    for tweet in sample_tweets:
        cleaned = preprocessor.clean_text(tweet)
        tokens = preprocessor.tokenize_text(cleaned)
        features = preprocessor.extract_features(tweet)
        
        print(f"Original: {tweet}")
        print(f"Cleaned: {cleaned}")
        print(f"Tokens: {tokens}")
        print(f"Features: {features}")
        print("-" * 50)


if __name__ == "__main__":
    main()

