"""
BERT model for sentiment analysis using HuggingFace transformers.

This module implements:
- BERT fine-tuning for sentiment classification
- DistilBERT for faster inference
- Custom training loop with advanced features
- Model optimization and conversion to TensorFlow Lite
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    BertTokenizer, BertForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentDataset(Dataset):
    """Custom dataset for sentiment analysis."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings
            labels: List of labels (0 or 1)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BertSentimentModel:
    """
    BERT-based sentiment analysis model with fine-tuning capabilities.
    """
    
    def __init__(
        self, 
        model_name: str = 'bert-base-uncased',
        model_dir: str = "models/",
        random_state: int = 42
    ):
        """
        Initialize BertSentimentModel.
        
        Args:
            model_name: Pre-trained model name ('bert-base-uncased', 'distilbert-base-uncased')
            model_dir: Directory to save model artifacts
            random_state: Random seed for reproducibility
        """
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_history = {}
        self.config = {}
        
        # Default configuration
        self.default_config = {
            'max_length': 128,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'num_epochs': 3,
            'warmup_steps': 500,
            'weight_decay': 0.01,
            'gradient_accumulation_steps': 1,
            'fp16': torch.cuda.is_available(),
            'save_strategy': 'epoch',
            'evaluation_strategy': 'epoch',
            'logging_steps': 100,
            'save_total_limit': 2,
            'load_best_model_at_end': True,
            'metric_for_best_model': 'f1',
            'greater_is_better': True,
            'early_stopping_patience': 2
        }
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def load_model_and_tokenizer(self) -> None:
        """Load pre-trained model and tokenizer."""
        logger.info(f"Loading {self.model_name} model and tokenizer...")
        
        try:
            # Load tokenizer
            if 'distilbert' in self.model_name.lower():
                self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=2,
                    output_attentions=False,
                    output_hidden_states=False
                )
            elif 'bert' in self.model_name.lower():
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                self.model = BertForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=2,
                    output_attentions=False,
                    output_hidden_states=False
                )
            else:
                # Generic AutoModel approach
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=2,
                    output_attentions=False,
                    output_hidden_states=False
                )
            
            # Move model to device
            self.model.to(self.device)
            
            logger.info(f"Model loaded successfully. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def prepare_datasets(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[SentimentDataset, Optional[SentimentDataset]]:
        """
        Prepare datasets for training.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts (optional)
            val_labels: Validation labels (optional)
            config: Configuration dictionary
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Update configuration
        self.config = {**self.default_config, **(config or {})}
        
        if self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        # Create training dataset
        train_dataset = SentimentDataset(
            train_texts, 
            train_labels, 
            self.tokenizer, 
            self.config['max_length']
        )
        
        # Create validation dataset if provided
        val_dataset = None
        if val_texts is not None and val_labels is not None:
            val_dataset = SentimentDataset(
                val_texts,
                val_labels,
                self.tokenizer,
                self.config['max_length']
            )
        
        logger.info(f"Datasets prepared - Train: {len(train_dataset)}, Val: {len(val_dataset) if val_dataset else 0}")
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            eval_pred: Evaluation predictions from trainer
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fine-tune BERT model for sentiment analysis.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            config: Training configuration
            
        Returns:
            Training history
        """
        logger.info("Starting BERT fine-tuning...")
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets(
            train_texts, train_labels, val_texts, val_labels, config
        )
        
        # Training arguments
        output_dir = self.model_dir / f"{self.model_name.replace('/', '_')}_finetuned"
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            warmup_steps=self.config['warmup_steps'],
            weight_decay=self.config['weight_decay'],
            learning_rate=self.config['learning_rate'],
            logging_dir=str(output_dir / 'logs'),
            logging_steps=self.config['logging_steps'],
            evaluation_strategy=self.config['evaluation_strategy'],
            save_strategy=self.config['save_strategy'],
            save_total_limit=self.config['save_total_limit'],
            load_best_model_at_end=self.config['load_best_model_at_end'],
            metric_for_best_model=self.config['metric_for_best_model'],
            greater_is_better=self.config['greater_is_better'],
            fp16=self.config['fp16'],
            dataloader_pin_memory=False,
            report_to=None  # Disable wandb/tensorboard logging
        )
        
        # Initialize trainer
        callbacks = []
        if self.config.get('early_stopping_patience'):
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=self.config['early_stopping_patience'])
            )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )
        
        # Start training
        logger.info(f"Training on {len(train_dataset)} samples...")
        train_result = self.trainer.train()
        
        # Save training history
        self.training_history = {
            'train_result': {
                'train_loss': train_result.training_loss,
                'train_runtime': train_result.metrics.get('train_runtime', 0),
                'train_samples_per_second': train_result.metrics.get('train_samples_per_second', 0)
            },
            'config': self.config,
            'model_name': self.model_name
        }
        
        # Evaluation on validation set if available
        if val_dataset is not None:
            eval_result = self.trainer.evaluate()
            self.training_history['eval_result'] = eval_result
            logger.info(f"Validation Results: {eval_result}")
        
        logger.info("Fine-tuning completed successfully")
        return self.training_history
    
    def predict(self, texts: List[str], batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new texts.
        
        Args:
            texts: List of texts to predict
            batch_size: Batch size for prediction
            
        Returns:
            Tuple of (predicted_labels, predicted_probabilities)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        batch_size = batch_size or self.config.get('batch_size', 16)
        
        # Create dataset
        dummy_labels = [0] * len(texts)  # Dummy labels for prediction
        dataset = SentimentDataset(texts, dummy_labels, self.tokenizer, self.config['max_length'])
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Make predictions
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Convert to probabilities
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)
    
    def evaluate(self, test_texts: List[str], test_labels: List[int]) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            test_texts: Test texts
            test_labels: Test labels
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model on test set...")
        
        # Get predictions
        y_pred, y_pred_proba = self.predict(test_texts)
        y_true = np.array(test_labels)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        accuracy = accuracy_score(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        logger.info(f"Test Results - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return metrics
    
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
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save the fine-tuned model.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            Path where model was saved
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("No trained model to save")
        
        if filepath is None:
            filepath = self.model_dir / f"{self.model_name.replace('/', '_')}_finetuned"
        else:
            filepath = Path(filepath)
        
        filepath.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(str(filepath))
        self.tokenizer.save_pretrained(str(filepath))
        
        # Save configuration and history
        config_path = filepath / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump({
                'config': self.config,
                'model_name': self.model_name,
                'training_history': self.training_history
            }, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
        return str(filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a fine-tuned model.
        
        Args:
            filepath: Path to the saved model
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model directory not found: {filepath}")
        
        logger.info(f"Loading model from {filepath}")
        
        # Load configuration
        config_path = filepath / 'training_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                data = json.load(f)
                self.config = data.get('config', {})
                self.model_name = data.get('model_name', self.model_name)
                self.training_history = data.get('training_history', {})
        
        # Load tokenizer and model
        if 'distilbert' in self.model_name.lower():
            self.tokenizer = DistilBertTokenizer.from_pretrained(str(filepath))
            self.model = DistilBertForSequenceClassification.from_pretrained(str(filepath))
        elif 'bert' in self.model_name.lower():
            self.tokenizer = BertTokenizer.from_pretrained(str(filepath))
            self.model = BertForSequenceClassification.from_pretrained(str(filepath))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(str(filepath))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(filepath))
        
        # Move model to device
        self.model.to(self.device)
        
        logger.info("Model loaded successfully")
    
    def convert_to_onnx(self, filepath: str, sample_text: str = "This is a sample text") -> None:
        """
        Convert model to ONNX format for optimized inference.
        
        Args:
            filepath: Path to save ONNX model
            sample_text: Sample text for tracing
        """
        try:
            import torch.onnx
            
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model not loaded")
            
            # Prepare sample input
            inputs = self.tokenizer(
                sample_text,
                return_tensors="pt",
                max_length=self.config['max_length'],
                padding="max_length",
                truncation=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Export to ONNX
            self.model.eval()
            torch.onnx.export(
                self.model,
                (inputs['input_ids'], inputs['attention_mask']),
                filepath,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size'},
                    'attention_mask': {0: 'batch_size'},
                    'logits': {0: 'batch_size'}
                }
            )
            
            logger.info(f"Model converted to ONNX format: {filepath}")
            
        except ImportError:
            logger.error("ONNX not available. Install with: pip install onnx")
        except Exception as e:
            logger.error(f"Error converting to ONNX: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Model information dictionary
        """
        if self.model is None:
            return {"status": "not_trained"}
        
        info = {
            "status": "trained",
            "model_name": self.model_name,
            "config": self.config,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device),
            "training_history": self.training_history
        }
        
        return info


def main():
    """Example usage of BertSentimentModel."""
    print("BertSentimentModel example usage:")
    print("1. Initialize model: model = BertSentimentModel('bert-base-uncased')")
    print("2. Train model: model.train(train_texts, train_labels, val_texts, val_labels)")
    print("3. Evaluate: metrics = model.evaluate(test_texts, test_labels)")
    print("4. Predict: predictions, probabilities = model.predict(new_texts)")
    print("5. Save model: model.save_model('path/to/model/')")


if __name__ == "__main__":
    main()













