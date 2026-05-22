"""
Logistic Regression model for sentiment analysis.

This module implements:
- Logistic regression with TF-IDF features
- Feature engineering and selection
- Cross-validation and hyperparameter tuning
- Model evaluation with comprehensive metrics
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogisticSentimentModel:
    """
    Logistic Regression model for sentiment analysis with comprehensive evaluation.
    """
    
    def __init__(self, model_dir: str = "models/", random_state: int = 42):
        """
        Initialize LogisticSentimentModel.
        
        Args:
            model_dir: Directory to save model artifacts
            random_state: Random seed for reproducibility
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        # Model components
        self.pipeline = None
        self.best_params = None
        self.feature_names = None
        self.training_history = {}
        
        # Default hyperparameters
        self.default_params = {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'liblinear',
            'max_iter': 1000,
            'random_state': random_state
        }
        
        # Hyperparameter search space
        self.param_grid = {
            'classifier__C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear'],
            'tfidf__max_features': [5000, 10000, 20000],
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'tfidf__min_df': [2, 3, 5],
            'tfidf__max_df': [0.8, 0.9, 0.95]
        }
    
    def create_pipeline(self, include_features: bool = True) -> Pipeline:
        """
        Create scikit-learn pipeline.
        
        Args:
            include_features: Whether to include hand-crafted features
            
        Returns:
            Configured pipeline
        """
        if include_features:
            # Create column transformer for text and numerical features
            preprocessor = ColumnTransformer([
                ('tfidf', TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    strip_accents='ascii',
                    lowercase=True,
                    stop_words='english'
                ), 'tokens_str'),
                ('features', StandardScaler(), [
                    'text_length', 'word_count', 'avg_word_length',
                    'uppercase_count', 'punctuation_count', 'digit_count',
                    'hashtag_count', 'mention_count', 'url_count', 'emoji_count',
                    'exclamation_count', 'question_count', 'caps_ratio', 'repeated_chars'
                ])
            ])
        else:
            # Text-only pipeline
            preprocessor = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9,
                strip_accents='ascii',
                lowercase=True,
                stop_words='english'
            )
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(**self.default_params))
        ])
        
        return pipeline
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        hyperparameter_tuning: bool = True,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train the logistic regression model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            cv_folds: Number of cross-validation folds
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting logistic regression training...")
        
        # Create pipeline
        self.pipeline = self.create_pipeline(include_features=True)
        
        if hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                self.pipeline,
                self.param_grid,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            self.pipeline = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
        else:
            # Train with default parameters
            self.pipeline.fit(X_train, y_train)
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            self.pipeline, X_train, y_train,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
            scoring='f1'
        )
        
        # Training predictions
        y_train_pred = self.pipeline.predict(X_train)
        y_train_pred_proba = self.pipeline.predict_proba(X_train)[:, 1]
        
        # Calculate training metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_pred_proba)
        
        # Validation metrics if provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            y_val_pred = self.pipeline.predict(X_val)
            y_val_pred_proba = self.pipeline.predict_proba(X_val)[:, 1]
            val_metrics = self._calculate_metrics(y_val, y_val_pred, y_val_pred_proba)
        
        # Store training history
        self.training_history = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_params': self.best_params,
            'feature_importance': self._get_feature_importance()
        }
        
        logger.info(f"Training completed. CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return self.training_history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.pipeline.predict_proba(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model on test set...")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate classification report
        metrics['classification_report'] = classification_report(y_test, y_pred)
        
        logger.info(f"Test Results - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if self.pipeline is None:
            return {}
        
        try:
            # Get the classifier coefficients
            classifier = self.pipeline.named_steps['classifier']
            coefficients = classifier.coef_[0]
            
            # Get feature names from TF-IDF vectorizer
            preprocessor = self.pipeline.named_steps['preprocessor']
            
            if hasattr(preprocessor, 'named_transformers_'):
                # ColumnTransformer case
                tfidf_transformer = preprocessor.named_transformers_['tfidf']
                feature_names = tfidf_transformer.get_feature_names_out()
                
                # Add hand-crafted feature names
                feature_names = list(feature_names) + [
                    'text_length', 'word_count', 'avg_word_length',
                    'uppercase_count', 'punctuation_count', 'digit_count',
                    'hashtag_count', 'mention_count', 'url_count', 'emoji_count',
                    'exclamation_count', 'question_count', 'caps_ratio', 'repeated_chars'
                ]
            else:
                # TfidfVectorizer case
                feature_names = preprocessor.get_feature_names_out()
            
            # Create feature importance dictionary
            feature_importance = dict(zip(feature_names, coefficients))
            
            # Sort by absolute importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            
            return dict(sorted_features[:50])  # Top 50 features
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
            return {}
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix - Logistic Regression')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save the plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Logistic Regression')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None) -> None:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
            save_path: Path to save the plot
        """
        if not self.training_history.get('feature_importance'):
            logger.warning("No feature importance data available")
            return
        
        feature_importance = self.training_history['feature_importance']
        
        # Get top N features by absolute importance
        top_features = list(feature_importance.items())[:top_n]
        features, importance = zip(*top_features)
        
        plt.figure(figsize=(10, 8))
        colors = ['red' if imp < 0 else 'blue' for imp in importance]
        bars = plt.barh(range(len(features)), importance, color=colors, alpha=0.7)
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Coefficient')
        plt.title(f'Top {top_n} Feature Importance - Logistic Regression')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, imp) in enumerate(zip(bars, importance)):
            plt.text(imp + (0.01 if imp > 0 else -0.01), i, f'{imp:.3f}', 
                    va='center', ha='left' if imp > 0 else 'right', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            Path where model was saved
        """
        if self.pipeline is None:
            raise ValueError("No trained model to save")
        
        if filepath is None:
            filepath = self.model_dir / 'logistic_regression_model.pkl'
        else:
            filepath = Path(filepath)
        
        # Save the complete pipeline
        joblib.dump(self.pipeline, filepath)
        
        # Save training history
        history_path = filepath.with_suffix('.history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)
        
        logger.info(f"Model saved to {filepath}")
        return str(filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load the pipeline
        self.pipeline = joblib.load(filepath)
        
        # Load training history if available
        history_path = filepath.with_suffix('.history.pkl')
        if history_path.exists():
            with open(history_path, 'rb') as f:
                self.training_history = pickle.load(f)
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Model information dictionary
        """
        if self.pipeline is None:
            return {"status": "not_trained"}
        
        info = {
            "status": "trained",
            "model_type": "LogisticRegression",
            "best_params": self.best_params,
            "training_history": self.training_history
        }
        
        return info


def main():
    """Example usage of LogisticSentimentModel."""
    # This would typically be called with real data
    print("LogisticSentimentModel example usage:")
    print("1. Initialize model: model = LogisticSentimentModel()")
    print("2. Train model: model.train(X_train, y_train)")
    print("3. Evaluate: metrics = model.evaluate(X_test, y_test)")
    print("4. Predict: predictions = model.predict(X_new)")
    print("5. Save model: model.save_model('path/to/model.pkl')")


if __name__ == "__main__":
    main()

