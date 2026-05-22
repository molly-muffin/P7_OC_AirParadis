"""
Deep Learning models for sentiment analysis.

This module implements:
- LSTM bidirectional with attention mechanism
- CNN 1D for feature extraction
- Hybrid CNN-LSTM architecture
- Support for Word2Vec and GloVe embeddings
- Hyperparameter optimization
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set TensorFlow logging level
tf.get_logger().setLevel(logging.ERROR)


class AttentionLayer(layers.Layer):
    """Custom attention layer for LSTM models."""
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_weight'
        )
        self.b = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True,
            name='attention_bias'
        )
        self.u = self.add_weight(
            shape=(input_shape[-1],),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_context'
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.nn.softmax(ait, axis=1)
        ait = tf.expand_dims(ait, axis=-1)
        weighted_input = inputs * ait
        output = tf.reduce_sum(weighted_input, axis=1)
        return output
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()


class DeepLearningModel:
    """
    Deep Learning models for sentiment analysis with multiple architectures.
    """
    
    def __init__(self, model_dir: str = "models/", random_state: int = 42):
        """
        Initialize DeepLearningModel.
        
        Args:
            model_dir: Directory to save model artifacts
            random_state: Random seed for reproducibility
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        # Set random seeds
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.model_type = None
        self.training_history = {}
        self.config = {}
        
        # Default configuration
        self.default_config = {
            'max_features': 20000,
            'max_length': 100,
            'embedding_dim': 300,
            'lstm_units': 128,
            'cnn_filters': 128,
            'kernel_size': 5,
            'dropout_rate': 0.5,
            'dense_units': 64,
            'batch_size': 32,
            'epochs': 10,
            'learning_rate': 0.001,
            'patience': 3
        }
    
    def prepare_data(
        self, 
        texts: List[str], 
        labels: Optional[np.ndarray] = None,
        fit_tokenizer: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare text data for deep learning models.
        
        Args:
            texts: List of text strings
            labels: Optional labels array
            fit_tokenizer: Whether to fit the tokenizer
            
        Returns:
            Tuple of (sequences, labels)
        """
        if fit_tokenizer or self.tokenizer is None:
            logger.info("Fitting tokenizer...")
            self.tokenizer = Tokenizer(
                num_words=self.config['max_features'],
                oov_token="<OOV>",
                filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
            )
            self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        X = pad_sequences(
            sequences, 
            maxlen=self.config['max_length'],
            padding='post',
            truncating='post'
        )
        
        return X, labels
    
    def create_lstm_model(self, embedding_matrix: Optional[np.ndarray] = None) -> Model:
        """
        Create bidirectional LSTM model with attention.
        
        Args:
            embedding_matrix: Pre-trained embeddings matrix
            
        Returns:
            Compiled Keras model
        """
        logger.info("Creating LSTM model with attention...")
        
        # Input layer
        input_layer = layers.Input(shape=(self.config['max_length'],))
        
        # Embedding layer
        if embedding_matrix is not None:
            embedding = layers.Embedding(
                input_dim=embedding_matrix.shape[0],
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                input_length=self.config['max_length'],
                trainable=False,
                mask_zero=True
            )(input_layer)
        else:
            embedding = layers.Embedding(
                input_dim=self.config['max_features'],
                output_dim=self.config['embedding_dim'],
                input_length=self.config['max_length'],
                mask_zero=True
            )(input_layer)
        
        # Bidirectional LSTM layers
        lstm_out = layers.Bidirectional(
            layers.LSTM(
                self.config['lstm_units'],
                return_sequences=True,
                dropout=self.config['dropout_rate'],
                recurrent_dropout=self.config['dropout_rate']
            )
        )(embedding)
        
        # Attention mechanism
        attention_out = AttentionLayer()(lstm_out)
        
        # Dense layers
        dense = layers.Dense(
            self.config['dense_units'],
            activation='relu'
        )(attention_out)
        dense = layers.Dropout(self.config['dropout_rate'])(dense)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid')(dense)
        
        # Create and compile model
        model = Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_cnn_model(self, embedding_matrix: Optional[np.ndarray] = None) -> Model:
        """
        Create 1D CNN model for text classification.
        
        Args:
            embedding_matrix: Pre-trained embeddings matrix
            
        Returns:
            Compiled Keras model
        """
        logger.info("Creating CNN model...")
        
        # Input layer
        input_layer = layers.Input(shape=(self.config['max_length'],))
        
        # Embedding layer
        if embedding_matrix is not None:
            embedding = layers.Embedding(
                input_dim=embedding_matrix.shape[0],
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                input_length=self.config['max_length'],
                trainable=False
            )(input_layer)
        else:
            embedding = layers.Embedding(
                input_dim=self.config['max_features'],
                output_dim=self.config['embedding_dim'],
                input_length=self.config['max_length']
            )(input_layer)
        
        # Multiple CNN layers with different kernel sizes
        conv_layers = []
        for kernel_size in [3, 4, 5]:
            conv = layers.Conv1D(
                filters=self.config['cnn_filters'],
                kernel_size=kernel_size,
                activation='relu'
            )(embedding)
            conv = layers.GlobalMaxPooling1D()(conv)
            conv_layers.append(conv)
        
        # Concatenate different conv layers
        if len(conv_layers) > 1:
            concat = layers.Concatenate()(conv_layers)
        else:
            concat = conv_layers[0]
        
        # Dense layers
        dense = layers.Dense(
            self.config['dense_units'],
            activation='relu'
        )(concat)
        dense = layers.Dropout(self.config['dropout_rate'])(dense)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid')(dense)
        
        # Create and compile model
        model = Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_hybrid_model(self, embedding_matrix: Optional[np.ndarray] = None) -> Model:
        """
        Create hybrid CNN-LSTM model.
        
        Args:
            embedding_matrix: Pre-trained embeddings matrix
            
        Returns:
            Compiled Keras model
        """
        logger.info("Creating hybrid CNN-LSTM model...")
        
        # Input layer
        input_layer = layers.Input(shape=(self.config['max_length'],))
        
        # Embedding layer
        if embedding_matrix is not None:
            embedding = layers.Embedding(
                input_dim=embedding_matrix.shape[0],
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                input_length=self.config['max_length'],
                trainable=False,
                mask_zero=True
            )(input_layer)
        else:
            embedding = layers.Embedding(
                input_dim=self.config['max_features'],
                output_dim=self.config['embedding_dim'],
                input_length=self.config['max_length'],
                mask_zero=True
            )(input_layer)
        
        # CNN layers for feature extraction
        conv = layers.Conv1D(
            filters=self.config['cnn_filters'],
            kernel_size=self.config['kernel_size'],
            activation='relu',
            padding='same'
        )(embedding)
        conv = layers.Dropout(self.config['dropout_rate'])(conv)
        
        # LSTM layers
        lstm_out = layers.Bidirectional(
            layers.LSTM(
                self.config['lstm_units'],
                return_sequences=True,
                dropout=self.config['dropout_rate']
            )
        )(conv)
        
        # Attention mechanism
        attention_out = AttentionLayer()(lstm_out)
        
        # Dense layers
        dense = layers.Dense(
            self.config['dense_units'],
            activation='relu'
        )(attention_out)
        dense = layers.Dropout(self.config['dropout_rate'])(dense)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid')(dense)
        
        # Create and compile model
        model = Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        model_type: str = 'lstm',
        embedding_matrix: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train the deep learning model.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            model_type: Type of model ('lstm', 'cnn', 'hybrid')
            embedding_matrix: Pre-trained embeddings
            config: Model configuration
            
        Returns:
            Training history
        """
        # Update configuration
        self.config = {**self.default_config, **(config or {})}
        self.model_type = model_type
        
        logger.info(f"Training {model_type} model...")
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val) if X_val is not None else 0}")
        
        # Create model based on type
        if model_type == 'lstm':
            self.model = self.create_lstm_model(embedding_matrix)
        elif model_type == 'cnn':
            self.model = self.create_cnn_model(embedding_matrix)
        elif model_type == 'hybrid':
            self.model = self.create_hybrid_model(embedding_matrix)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Model created with {self.model.count_params()} parameters")
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.config['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Training
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=1
        )
        
        # Store training history
        self.training_history = {
            'history': history.history,
            'config': self.config,
            'model_type': model_type
        }
        
        logger.info("Training completed")
        return self.training_history
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input sequences
            batch_size: Batch size for prediction
            
        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X, batch_size=batch_size)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test sequences
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model on test set...")
        
        # Get predictions
        y_pred_proba = self.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_pred_proba = y_pred_proba.flatten()
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Model evaluation
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        metrics['test_loss'] = test_loss
        metrics['test_accuracy_keras'] = test_accuracy
        
        logger.info(f"Test Results - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.training_history.get('history'):
            logger.warning("No training history available")
            return
        
        history = self.training_history['history']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history:
            axes[0].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[1].plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            axes[1].plot(history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
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
        plt.title(f'Confusion Matrix - {self.model_type.upper()} Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
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
        if self.model is None:
            raise ValueError("No trained model to save")
        
        if filepath is None:
            filepath = self.model_dir / f'{self.model_type}_model.h5'
        else:
            filepath = Path(filepath)
        
        # Save the Keras model
        self.model.save(filepath)
        
        # Save tokenizer
        tokenizer_path = filepath.with_suffix('.tokenizer.pkl')
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # Save configuration and history
        config_path = filepath.with_suffix('.config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'config': self.config,
                'model_type': self.model_type,
                'training_history': self.training_history
            }, f, indent=2)
        
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
        
        # Load the Keras model
        custom_objects = {'AttentionLayer': AttentionLayer}
        self.model = keras.models.load_model(filepath, custom_objects=custom_objects)
        
        # Load tokenizer
        tokenizer_path = filepath.with_suffix('.tokenizer.pkl')
        if tokenizer_path.exists():
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
        
        # Load configuration
        config_path = filepath.with_suffix('.config.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                data = json.load(f)
                self.config = data.get('config', {})
                self.model_type = data.get('model_type', 'unknown')
                self.training_history = data.get('training_history', {})
        
        logger.info(f"Model loaded from {filepath}")
    
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
            "model_type": self.model_type,
            "config": self.config,
            "parameters": self.model.count_params(),
            "training_history": self.training_history
        }
        
        return info


def main():
    """Example usage of DeepLearningModel."""
    print("DeepLearningModel example usage:")
    print("1. Initialize model: model = DeepLearningModel()")
    print("2. Prepare data: X_train, y_train = model.prepare_data(texts, labels, fit_tokenizer=True)")
    print("3. Train model: model.train(X_train, y_train, model_type='lstm')")
    print("4. Evaluate: metrics = model.evaluate(X_test, y_test)")
    print("5. Save model: model.save_model('path/to/model.h5')")


if __name__ == "__main__":
    main()

