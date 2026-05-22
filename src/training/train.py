"""
Training script for sentiment analysis models with MLflow tracking.

This script provides a unified interface for training all model types:
- Logistic Regression
- Deep Learning (LSTM, CNN, Hybrid)
- BERT/DistilBERT

Features:
- MLflow experiment tracking
- Hyperparameter tuning
- Cross-validation
- Model comparison
- Automated model registration
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import DataLoader
from data.preprocessor import TextPreprocessor
from models.logistic_model import LogisticSentimentModel
from models.deep_learning_model import DeepLearningModel
from models.bert_model import BertSentimentModel
sys.path.append(str(Path(__file__).parent.parent.parent / "mlflow"))
from mlflow_config import MLflowTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Unified model trainer with MLflow tracking.
    """
    
    def __init__(
        self,
        data_path: str = "data/",
        model_dir: str = "models/",
        experiment_name: str = "air-paradis-sentiment"
    ):
        """
        Initialize ModelTrainer.
        
        Args:
            data_path: Path to data directory
            model_dir: Path to model directory
            experiment_name: MLflow experiment name
        """
        self.data_path = data_path
        self.model_dir = model_dir
        
        # Initialize components
        self.data_loader = DataLoader(data_path)
        self.preprocessor = TextPreprocessor()
        self.mlflow_tracker = MLflowTracker(experiment_name)
        
        # Data containers
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        # Model configurations
        self.model_configs = {
            'logistic': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'max_features': [5000, 10000, 20000]
            },
            'lstm': {
                'lstm_units': [64, 128, 256],
                'dropout_rate': [0.3, 0.5, 0.7],
                'learning_rate': [0.001, 0.01],
                'epochs': [10, 15, 20]
            },
            'cnn': {
                'cnn_filters': [64, 128, 256],
                'kernel_size': [3, 5, 7],
                'dropout_rate': [0.3, 0.5, 0.7],
                'epochs': [10, 15, 20]
            },
            'hybrid': {
                'lstm_units': [64, 128],
                'cnn_filters': [64, 128],
                'dropout_rate': [0.3, 0.5],
                'epochs': [10, 15, 20]
            },
            'bert': {
                'learning_rate': [2e-5, 3e-5, 5e-5],
                'batch_size': [16, 32],
                'num_epochs': [3, 4, 5],
                'warmup_steps': [500, 1000]
            }
        }
    
    def load_and_preprocess_data(self, sample_size: Optional[int] = None) -> None:
        """
        Load and preprocess data.
        
        Args:
            sample_size: Optional sample size for testing
        """
        logger.info("Loading and preprocessing data...")
        
        try:
            # Load processed data or process raw data
            try:
                data_splits = self.data_loader.load_processed_data('all')
                logger.info("Loaded cached processed data")
            except FileNotFoundError:
                logger.info("No cached data found. Processing raw data...")
                data_splits = self.data_loader.process_data()
            
            # Sample data if requested
            if sample_size:
                logger.info(f"Sampling {sample_size} examples for testing")
                for split in ['train', 'val', 'test']:
                    if split in data_splits:
                        n_samples = min(sample_size, len(data_splits[split]))
                        data_splits[split] = data_splits[split].sample(n=n_samples, random_state=42)
            
            # Preprocess text
            for split_name, df in data_splits.items():
                if split_name != 'full':
                    logger.info(f"Preprocessing {split_name} data...")
                    processed_df = self.preprocessor.preprocess_dataframe(df)
                    data_splits[split_name] = processed_df
            
            self.train_data = data_splits['train']
            self.val_data = data_splits['val']
            self.test_data = data_splits['test']
            
            logger.info(f"Data loaded - Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def train_logistic_model(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train logistic regression model.
        
        Args:
            config: Model configuration
            
        Returns:
            Training results
        """
        logger.info("Training Logistic Regression model...")
        
        # Start MLflow run
        run_id = self.mlflow_tracker.start_run(
            run_name="logistic_regression",
            tags={"model_type": "logistic_regression", "framework": "sklearn"}
        )
        
        try:
            # Initialize model
            model = LogisticSentimentModel(model_dir=self.model_dir)
            
            # Prepare data
            X_train = self.train_data
            y_train = self.train_data['target']
            X_val = self.val_data
            y_val = self.val_data['target']
            
            # Log parameters
            model_config = config or {}
            self.mlflow_tracker.log_params({
                'model_type': 'logistic_regression',
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                **model_config
            })
            
            # Train model
            training_results = model.train(
                X_train, y_train, X_val, y_val,
                hyperparameter_tuning=True
            )
            
            # Log training metrics
            if 'train_metrics' in training_results:
                train_metrics = {f"train_{k}": v for k, v in training_results['train_metrics'].items() if isinstance(v, (int, float))}
                self.mlflow_tracker.log_metrics(train_metrics)
            
            if 'val_metrics' in training_results:
                val_metrics = {f"val_{k}": v for k, v in training_results['val_metrics'].items() if isinstance(v, (int, float))}
                self.mlflow_tracker.log_metrics(val_metrics)
            
            # Log CV scores
            if 'cv_mean' in training_results:
                self.mlflow_tracker.log_metrics({
                    'cv_f1_mean': training_results['cv_mean'],
                    'cv_f1_std': training_results['cv_std']
                })
            
            # Evaluate on test set
            test_metrics = model.evaluate(self.test_data, self.test_data['target'])
            test_metrics_logged = {f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))}
            self.mlflow_tracker.log_metrics(test_metrics_logged)
            
            # Log model
            self.mlflow_tracker.log_model(model.pipeline, 'sklearn')
            
            # Log artifacts
            y_test_pred = model.predict(self.test_data)
            self.mlflow_tracker.log_confusion_matrix(self.test_data['target'], y_test_pred)
            self.mlflow_tracker.log_classification_report(self.test_data['target'], y_test_pred)
            
            # Save model
            model_path = model.save_model()
            
            logger.info(f"Logistic Regression training completed. Test F1: {test_metrics['f1']:.4f}")
            
            return {
                'model': model,
                'metrics': test_metrics,
                'model_path': model_path,
                'run_id': run_id
            }
            
        except Exception as e:
            logger.error(f"Error training logistic model: {str(e)}")
            raise
        finally:
            self.mlflow_tracker.end_run()
    
    def train_deep_learning_model(
        self, 
        model_type: str = 'lstm',
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train deep learning model.
        
        Args:
            model_type: Type of model ('lstm', 'cnn', 'hybrid')
            config: Model configuration
            
        Returns:
            Training results
        """
        logger.info(f"Training {model_type.upper()} model...")
        
        # Start MLflow run
        run_id = self.mlflow_tracker.start_run(
            run_name=f"{model_type}_model",
            tags={"model_type": model_type, "framework": "tensorflow"}
        )
        
        try:
            # Initialize model
            model = DeepLearningModel(model_dir=self.model_dir)
            
            # Prepare data
            train_texts = self.train_data['tokens_str'].tolist()
            train_labels = self.train_data['target'].values
            val_texts = self.val_data['tokens_str'].tolist()
            val_labels = self.val_data['target'].values
            
            # Prepare sequences
            X_train, y_train = model.prepare_data(train_texts, train_labels, fit_tokenizer=True)
            X_val, y_val = model.prepare_data(val_texts, val_labels, fit_tokenizer=False)
            
            # Train word embeddings
            tokenized_texts = self.train_data['tokens'].tolist()
            model.train_word2vec(tokenized_texts)
            
            # Get embeddings matrix
            vocab = list(model.tokenizer.word_index.keys())
            embeddings_matrix = model.get_embeddings_matrix(vocab[:model.config.get('max_features', 20000)])
            
            # Log parameters
            model_config = config or {}
            self.mlflow_tracker.log_params({
                'model_type': model_type,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'vocab_size': len(vocab),
                **model_config
            })
            
            # Train model
            training_results = model.train(
                X_train, y_train, X_val, y_val,
                model_type=model_type,
                embedding_matrix=embeddings_matrix,
                config=model_config
            )
            
            # Log training history
            if 'history' in training_results:
                self.mlflow_tracker.log_training_history(training_results['history'])
            
            # Evaluate on test set
            test_texts = self.test_data['tokens_str'].tolist()
            test_labels = self.test_data['target'].values
            X_test, y_test = model.prepare_data(test_texts, test_labels, fit_tokenizer=False)
            
            test_metrics = model.evaluate(X_test, y_test)
            test_metrics_logged = {f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))}
            self.mlflow_tracker.log_metrics(test_metrics_logged)
            
            # Log model
            self.mlflow_tracker.log_model(model.model, 'tensorflow')
            
            # Log artifacts
            y_test_pred = model.predict(X_test)
            y_test_pred_binary = (y_test_pred > 0.5).astype(int).flatten()
            self.mlflow_tracker.log_confusion_matrix(y_test, y_test_pred_binary)
            self.mlflow_tracker.log_classification_report(y_test, y_test_pred_binary)
            
            # Save model
            model_path = model.save_model()
            
            logger.info(f"{model_type.upper()} training completed. Test F1: {test_metrics['f1']:.4f}")
            
            return {
                'model': model,
                'metrics': test_metrics,
                'model_path': model_path,
                'run_id': run_id
            }
            
        except Exception as e:
            logger.error(f"Error training {model_type} model: {str(e)}")
            raise
        finally:
            self.mlflow_tracker.end_run()
    
    def train_bert_model(
        self,
        model_name: str = 'distilbert-base-uncased',
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train BERT model.
        
        Args:
            model_name: Pre-trained model name
            config: Model configuration
            
        Returns:
            Training results
        """
        logger.info(f"Training {model_name} model...")
        
        # Start MLflow run
        run_id = self.mlflow_tracker.start_run(
            run_name=f"bert_{model_name.replace('/', '_')}",
            tags={"model_type": "bert", "framework": "transformers", "base_model": model_name}
        )
        
        try:
            # Initialize model
            model = BertSentimentModel(model_name=model_name, model_dir=self.model_dir)
            
            # Prepare data
            train_texts = self.train_data['cleaned_text'].tolist()
            train_labels = self.train_data['target'].tolist()
            val_texts = self.val_data['cleaned_text'].tolist()
            val_labels = self.val_data['target'].tolist()
            
            # Log parameters
            model_config = config or {}
            self.mlflow_tracker.log_params({
                'model_type': 'bert',
                'base_model': model_name,
                'train_samples': len(train_texts),
                'val_samples': len(val_texts),
                **model_config
            })
            
            # Train model
            training_results = model.train(
                train_texts, train_labels,
                val_texts, val_labels,
                config=model_config
            )
            
            # Log training metrics
            if 'train_result' in training_results:
                train_metrics = {f"train_{k}": v for k, v in training_results['train_result'].items() if isinstance(v, (int, float))}
                self.mlflow_tracker.log_metrics(train_metrics)
            
            if 'eval_result' in training_results:
                eval_metrics = {f"val_{k}": v for k, v in training_results['eval_result'].items() if isinstance(v, (int, float))}
                self.mlflow_tracker.log_metrics(eval_metrics)
            
            # Evaluate on test set
            test_texts = self.test_data['cleaned_text'].tolist()
            test_labels = self.test_data['target'].tolist()
            
            test_metrics = model.evaluate(test_texts, test_labels)
            test_metrics_logged = {f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))}
            self.mlflow_tracker.log_metrics(test_metrics_logged)
            
            # Log artifacts
            y_test_pred, _ = model.predict(test_texts)
            self.mlflow_tracker.log_confusion_matrix(test_labels, y_test_pred)
            self.mlflow_tracker.log_classification_report(test_labels, y_test_pred)
            
            # Save model
            model_path = model.save_model()
            
            logger.info(f"BERT training completed. Test F1: {test_metrics['f1']:.4f}")
            
            return {
                'model': model,
                'metrics': test_metrics,
                'model_path': model_path,
                'run_id': run_id
            }
            
        except Exception as e:
            logger.error(f"Error training BERT model: {str(e)}")
            raise
        finally:
            self.mlflow_tracker.end_run()
    
    def train_all_models(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Train all model types and compare results.
        
        Args:
            sample_size: Optional sample size for testing
            
        Returns:
            Results from all models
        """
        logger.info("Training all models...")
        
        # Load data
        self.load_and_preprocess_data(sample_size)
        
        results = {}
        
        try:
            # Train Logistic Regression
            results['logistic'] = self.train_logistic_model()
            
            # Train Deep Learning models
            for model_type in ['lstm', 'cnn', 'hybrid']:
                results[model_type] = self.train_deep_learning_model(model_type)
            
            # Train BERT models
            for model_name in ['distilbert-base-uncased', 'bert-base-uncased']:
                key = f"bert_{model_name.replace('/', '_').replace('-', '_')}"
                results[key] = self.train_bert_model(model_name)
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise
        
        # Compare results
        self._compare_models(results)
        
        return results
    
    def _compare_models(self, results: Dict[str, Any]) -> None:
        """
        Compare model results and log comparison.
        
        Args:
            results: Dictionary of model results
        """
        logger.info("Comparing model results...")
        
        comparison_data = []
        for model_name, result in results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                comparison_data.append({
                    'model': model_name,
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1': metrics.get('f1', 0),
                    'run_id': result.get('run_id', '')
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('f1', ascending=False)
            
            logger.info("Model Comparison Results:")
            logger.info(f"\n{comparison_df.to_string(index=False)}")
            
            # Find best model
            best_model = comparison_df.iloc[0]
            logger.info(f"Best model: {best_model['model']} (F1: {best_model['f1']:.4f})")
            
            # Register best model
            if best_model['run_id']:
                try:
                    self.mlflow_tracker.register_model(
                        model_name="air_paradis_sentiment_best",
                        description=f"Best performing model: {best_model['model']} with F1 score: {best_model['f1']:.4f}"
                    )
                except Exception as e:
                    logger.warning(f"Could not register best model: {str(e)}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train sentiment analysis models')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'logistic', 'lstm', 'cnn', 'hybrid', 'bert'],
                       help='Model type to train')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Sample size for testing (optional)')
    parser.add_argument('--data-path', type=str, default='data/',
                       help='Path to data directory')
    parser.add_argument('--model-dir', type=str, default='models/',
                       help='Path to model directory')
    parser.add_argument('--experiment-name', type=str, default='air-paradis-sentiment',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer(
        data_path=args.data_path,
        model_dir=args.model_dir,
        experiment_name=args.experiment_name
    )
    
    # Load data
    trainer.load_and_preprocess_data(args.sample_size)
    
    try:
        if args.model == 'all':
            results = trainer.train_all_models(args.sample_size)
        elif args.model == 'logistic':
            results = trainer.train_logistic_model()
        elif args.model in ['lstm', 'cnn', 'hybrid']:
            results = trainer.train_deep_learning_model(args.model)
        elif args.model == 'bert':
            results = trainer.train_bert_model()
        else:
            raise ValueError(f"Unknown model type: {args.model}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    from pipeline import main as pipeline_main

    pipeline_main()













