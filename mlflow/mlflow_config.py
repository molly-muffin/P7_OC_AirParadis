"""
MLflow configuration and experiment tracking for Air Paradis sentiment analysis.

This module provides:
- MLflow experiment setup and configuration
- Model tracking and versioning
- Metrics and artifact logging
- Model registry management
- Experiment comparison utilities
"""

import os
import logging
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.pytorch
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
import json
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings

# Suppress MLflow warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    MLflow experiment tracking and model management for sentiment analysis.
    """
    
    def __init__(
        self, 
        experiment_name: str = "air-paradis-sentiment",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
            artifact_location: Location to store artifacts
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Default to local file store
            mlflow_dir = Path("mlruns")
            mlflow_dir.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location
                )
                logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
            
            mlflow.set_experiment(experiment_name)
            self.experiment_id = experiment_id
            
        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {str(e)}")
            raise
        
        self.current_run = None
        self.run_id = None
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Tags to add to the run
            
        Returns:
            Run ID
        """
        if self.current_run is not None:
            logger.warning("A run is already active. Ending it before starting a new one.")
            self.end_run()
        
        self.current_run = mlflow.start_run(run_name=run_name, tags=tags)
        self.run_id = self.current_run.info.run_id
        
        logger.info(f"Started MLflow run: {run_name or self.run_id}")
        return self.run_id
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        if self.current_run is not None:
            mlflow.end_run()
            self.current_run = None
            self.run_id = None
            logger.info("Ended MLflow run")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters to log
        """
        if self.current_run is None:
            logger.warning("No active run. Parameters will not be logged.")
            return
        
        # Convert complex types to strings
        processed_params = {}
        for key, value in params.items():
            if isinstance(value, (dict, list)):
                processed_params[key] = json.dumps(value)
            elif value is None:
                processed_params[key] = "None"
            else:
                processed_params[key] = str(value)
        
        mlflow.log_params(processed_params)
        logger.debug(f"Logged {len(processed_params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if self.current_run is None:
            logger.warning("No active run. Metrics will not be logged.")
            return
        
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged {len(metrics)} metrics")
    
    def log_model(
        self, 
        model: Any, 
        model_type: str, 
        artifact_path: str = "model",
        **kwargs
    ) -> None:
        """
        Log model to MLflow.
        
        Args:
            model: Model object to log
            model_type: Type of model ('sklearn', 'tensorflow', 'pytorch', 'transformers')
            artifact_path: Path within the run's artifact directory
            **kwargs: Additional arguments for model logging
        """
        if self.current_run is None:
            logger.warning("No active run. Model will not be logged.")
            return
        
        try:
            if model_type == 'sklearn':
                mlflow.sklearn.log_model(model, artifact_path, **kwargs)
            elif model_type == 'tensorflow':
                mlflow.tensorflow.log_model(model, artifact_path, **kwargs)
            elif model_type == 'pytorch':
                mlflow.pytorch.log_model(model, artifact_path, **kwargs)
            elif model_type == 'transformers':
                # For HuggingFace transformers, save as artifacts
                with tempfile.TemporaryDirectory() as tmp_dir:
                    model_path = Path(tmp_dir) / "model"
                    if hasattr(model, 'save_pretrained'):
                        model.save_pretrained(str(model_path))
                        mlflow.log_artifacts(str(model_path), artifact_path)
                    else:
                        logger.warning("Model does not have save_pretrained method")
            else:
                logger.warning(f"Unknown model type: {model_type}")
            
            logger.info(f"Logged {model_type} model to {artifact_path}")
            
        except Exception as e:
            logger.error(f"Error logging model: {str(e)}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log artifact to MLflow.
        
        Args:
            local_path: Path to local file/directory
            artifact_path: Path within the run's artifact directory
        """
        if self.current_run is None:
            logger.warning("No active run. Artifact will not be logged.")
            return
        
        mlflow.log_artifact(local_path, artifact_path)
        logger.debug(f"Logged artifact: {local_path}")
    
    def log_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        labels: List[str] = None
    ) -> None:
        """
        Log confusion matrix as artifact.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
        """
        if labels is None:
            labels = ['Negative', 'Positive']
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            plt.savefig(tmp_file.name, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log as artifact
            self.log_artifact(tmp_file.name, "plots/confusion_matrix.png")
        
        # Clean up
        os.unlink(tmp_file.name)
    
    def log_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Log classification report as artifact.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        report = classification_report(y_true, y_pred)
        
        # Save report as text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(report)
            tmp_file.flush()
            
            # Log as artifact
            self.log_artifact(tmp_file.name, "reports/classification_report.txt")
        
        # Clean up
        os.unlink(tmp_file.name)
    
    def log_training_history(self, history: Dict[str, List[float]]) -> None:
        """
        Log training history (for deep learning models).
        
        Args:
            history: Training history dictionary
        """
        # Log final metrics
        final_metrics = {}
        for metric_name, values in history.items():
            if isinstance(values, list) and len(values) > 0:
                final_metrics[f"final_{metric_name}"] = values[-1]
                
                # Log metrics for each epoch
                for epoch, value in enumerate(values):
                    self.log_metrics({metric_name: value}, step=epoch)
        
        self.log_metrics(final_metrics)
        
        # Create training curves plot
        if len(history) > 0:
            self._plot_training_curves(history)
    
    def _plot_training_curves(self, history: Dict[str, List[float]]) -> None:
        """
        Plot and log training curves.
        
        Args:
            history: Training history dictionary
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        if 'accuracy' in history:
            axes[0].plot(history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history:
            axes[0].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss
        if 'loss' in history:
            axes[1].plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            axes[1].plot(history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save and log plot
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            plt.savefig(tmp_file.name, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_artifact(tmp_file.name, "plots/training_curves.png")
        
        # Clean up
        os.unlink(tmp_file.name)
    
    def register_model(
        self, 
        model_name: str, 
        model_version: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register model in MLflow Model Registry.
        
        Args:
            model_name: Name for the registered model
            model_version: Version of the model
            description: Description of the model
            tags: Tags for the model
            
        Returns:
            Model version
        """
        if self.current_run is None:
            logger.warning("No active run. Cannot register model.")
            return None
        
        try:
            model_uri = f"runs:/{self.run_id}/model"
            
            result = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=tags
            )
            
            # Add description if provided
            if description:
                client = mlflow.tracking.MlflowClient()
                client.update_model_version(
                    name=model_name,
                    version=result.version,
                    description=description
                )
            
            logger.info(f"Registered model {model_name} version {result.version}")
            return result.version
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            return None
    
    def get_best_run(self, metric_name: str = "f1", ascending: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get the best run based on a metric.
        
        Args:
            metric_name: Metric to optimize
            ascending: Whether to sort in ascending order
            
        Returns:
            Best run information
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
            )
            
            if len(runs) > 0:
                best_run = runs.iloc[0]
                return {
                    'run_id': best_run['run_id'],
                    'metrics': {col.replace('metrics.', ''): best_run[col] 
                              for col in best_run.index if col.startswith('metrics.')},
                    'params': {col.replace('params.', ''): best_run[col] 
                             for col in best_run.index if col.startswith('params.')}
                }
            
        except Exception as e:
            logger.error(f"Error getting best run: {str(e)}")
        
        return None
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            DataFrame with run comparison
        """
        try:
            runs_data = []
            
            for run_id in run_ids:
                run = mlflow.get_run(run_id)
                run_data = {
                    'run_id': run_id,
                    'run_name': run.data.tags.get('mlflow.runName', 'Unnamed'),
                    'status': run.info.status,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time
                }
                
                # Add metrics
                run_data.update({f"metric_{k}": v for k, v in run.data.metrics.items()})
                
                # Add key parameters
                run_data.update({f"param_{k}": v for k, v in run.data.params.items()})
                
                runs_data.append(run_data)
            
            return pd.DataFrame(runs_data)
            
        except Exception as e:
            logger.error(f"Error comparing runs: {str(e)}")
            return pd.DataFrame()
    
    def load_model(self, run_id: str, artifact_path: str = "model") -> Any:
        """
        Load model from MLflow run.
        
        Args:
            run_id: MLflow run ID
            artifact_path: Path to model artifact
            
        Returns:
            Loaded model
        """
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            
            # Try different model flavors
            try:
                return mlflow.sklearn.load_model(model_uri)
            except:
                try:
                    return mlflow.tensorflow.load_model(model_uri)
                except:
                    try:
                        return mlflow.pytorch.load_model(model_uri)
                    except:
                        logger.error("Could not load model with any supported flavor")
                        return None
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def get_tracking_uri(self) -> str:
        """Get the current MLflow tracking URI."""
        return mlflow.get_tracking_uri()
    
    def get_experiment_info(self) -> Dict[str, Any]:
        """
        Get experiment information.
        
        Returns:
            Experiment information dictionary
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            
            return {
                'experiment_id': experiment.experiment_id,
                'experiment_name': experiment.name,
                'artifact_location': experiment.artifact_location,
                'lifecycle_stage': experiment.lifecycle_stage,
                'total_runs': len(runs),
                'tracking_uri': self.get_tracking_uri()
            }
            
        except Exception as e:
            logger.error(f"Error getting experiment info: {str(e)}")
            return {}


def setup_mlflow_tracking(
    experiment_name: str = "air-paradis-sentiment",
    tracking_uri: Optional[str] = None
) -> MLflowTracker:
    """
    Setup MLflow tracking for the project.
    
    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: MLflow tracking server URI
        
    Returns:
        Configured MLflowTracker instance
    """
    return MLflowTracker(experiment_name=experiment_name, tracking_uri=tracking_uri)


def main():
    """Example usage of MLflowTracker."""
    # Initialize tracker
    tracker = setup_mlflow_tracking()
    
    # Start a run
    tracker.start_run(run_name="example_run")
    
    # Log parameters
    tracker.log_params({
        'model_type': 'logistic_regression',
        'max_features': 10000,
        'C': 1.0
    })
    
    # Log metrics
    tracker.log_metrics({
        'accuracy': 0.85,
        'f1': 0.83,
        'precision': 0.86,
        'recall': 0.81
    })
    
    # End run
    tracker.end_run()
    
    # Get experiment info
    info = tracker.get_experiment_info()
    print("Experiment info:", info)


if __name__ == "__main__":
    main()













