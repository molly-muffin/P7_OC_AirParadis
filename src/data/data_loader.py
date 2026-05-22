"""
Data loading module for Sentiment140 dataset.

This module handles:
- Loading the Sentiment140 dataset (1.6M tweets)
- Data validation and cleaning
- Train/validation/test splitting (70/15/15)
- Optimized data storage and retrieval
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader for Sentiment140 dataset with preprocessing and splitting capabilities.
    """
    
    def __init__(self, data_path: str = "data/", cache_dir: str = "data/cache/"):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to raw data directory
            cache_dir: Path to cache processed data
        """
        self.data_path = Path(data_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Sentiment140 dataset column names
        self.columns = [
            'target', 'ids', 'date', 'flag', 'user', 'text'
        ]
        
        # Cache file paths
        self.cache_files = {
            'full_data': self.cache_dir / 'full_dataset.pkl',
            'train': self.cache_dir / 'train_data.pkl',
            'val': self.cache_dir / 'val_data.pkl',
            'test': self.cache_dir / 'test_data.pkl',
            'metadata': self.cache_dir / 'dataset_metadata.pkl'
        }
    
    def load_raw_data(self, filename: str = "training.1600000.processed.noemoticon.csv") -> pd.DataFrame:
        """
        Load raw Sentiment140 dataset.
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            DataFrame with loaded data
        """
        file_path = self.data_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {file_path}. "
                f"Please download the Sentiment140 dataset and place it in {self.data_path}"
            )
        
        logger.info(f"Loading raw data from {file_path}")
        
        try:
            # Load with appropriate encoding and column names
            df = pd.read_csv(
                file_path,
                encoding='latin-1',
                names=self.columns,
                dtype={
                    'target': 'int8',
                    'ids': 'int64',
                    'date': 'str',
                    'flag': 'str',
                    'user': 'str',
                    'text': 'str'
                }
            )
            
            logger.info(f"Loaded {len(df)} tweets")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the loaded data.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Validated and cleaned dataframe
        """
        logger.info("Validating data...")
        
        initial_size = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['text'])
        logger.info(f"Removed {initial_size - len(df)} duplicate tweets")
        
        # Remove null values
        df = df.dropna(subset=['text', 'target'])
        logger.info(f"Removed tweets with null text or target")
        
        # Convert sentiment labels (0=negative, 4=positive) to (0=negative, 1=positive)
        df['target'] = df['target'].map({0: 0, 4: 1})
        
        # Remove tweets that are too short (less than 3 characters)
        df = df[df['text'].str.len() >= 3]
        
        # Basic text cleaning
        df['text'] = df['text'].str.strip()
        
        logger.info(f"Final dataset size: {len(df)} tweets")
        logger.info(f"Sentiment distribution:\n{df['target'].value_counts()}")
        
        return df
    
    def create_splits(
        self, 
        df: pd.DataFrame, 
        train_size: float = 0.7, 
        val_size: float = 0.15, 
        test_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets.
        
        Args:
            df: Full dataset
            train_size: Proportion for training
            val_size: Proportion for validation
            test_size: Proportion for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split sizes must sum to 1.0"
        
        logger.info(f"Creating splits: train={train_size}, val={val_size}, test={test_size}")
        
        # First split: separate train from temp (val + test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=(val_size + test_size),
            random_state=random_state,
            stratify=df['target']
        )
        
        # Second split: separate val from test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_size / (val_size + test_size),
            random_state=random_state,
            stratify=temp_df['target']
        )
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_processed_data(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame,
        full_df: pd.DataFrame
    ) -> None:
        """
        Save processed data to cache.
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            full_df: Full dataset
        """
        logger.info("Saving processed data to cache...")
        
        # Save datasets
        with open(self.cache_files['train'], 'wb') as f:
            pickle.dump(train_df, f)
        
        with open(self.cache_files['val'], 'wb') as f:
            pickle.dump(val_df, f)
        
        with open(self.cache_files['test'], 'wb') as f:
            pickle.dump(test_df, f)
        
        with open(self.cache_files['full_data'], 'wb') as f:
            pickle.dump(full_df, f)
        
        # Save metadata
        metadata = {
            'total_samples': len(full_df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'sentiment_distribution': full_df['target'].value_counts().to_dict(),
            'columns': list(full_df.columns)
        }
        
        with open(self.cache_files['metadata'], 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info("Data saved to cache successfully")
    
    def load_processed_data(self, split: str = 'all') -> Dict[str, pd.DataFrame]:
        """
        Load processed data from cache.
        
        Args:
            split: Which split to load ('train', 'val', 'test', 'all')
            
        Returns:
            Dictionary containing requested data splits
        """
        if not self.cache_files['metadata'].exists():
            raise FileNotFoundError("No cached data found. Run process_data() first.")
        
        data = {}
        
        if split == 'all' or split == 'train':
            with open(self.cache_files['train'], 'rb') as f:
                data['train'] = pickle.load(f)
        
        if split == 'all' or split == 'val':
            with open(self.cache_files['val'], 'rb') as f:
                data['val'] = pickle.load(f)
        
        if split == 'all' or split == 'test':
            with open(self.cache_files['test'], 'rb') as f:
                data['test'] = pickle.load(f)
        
        if split == 'full':
            with open(self.cache_files['full_data'], 'rb') as f:
                data['full'] = pickle.load(f)
        
        return data
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get dataset metadata.
        
        Returns:
            Dictionary containing dataset metadata
        """
        if not self.cache_files['metadata'].exists():
            raise FileNotFoundError("No metadata found. Run process_data() first.")
        
        with open(self.cache_files['metadata'], 'rb') as f:
            return pickle.load(f)
    
    def process_data(self, filename: str = "training.1600000.processed.noemoticon.csv") -> Dict[str, pd.DataFrame]:
        """
        Complete data processing pipeline.
        
        Args:
            filename: Name of the raw data file
            
        Returns:
            Dictionary containing all data splits
        """
        logger.info("Starting data processing pipeline...")
        
        # Load raw data
        df = self.load_raw_data(filename)
        
        # Validate and clean
        df = self.validate_data(df)
        
        # Create splits
        train_df, val_df, test_df = self.create_splits(df)
        
        # Save processed data
        self.save_processed_data(train_df, val_df, test_df, df)
        
        logger.info("Data processing pipeline completed successfully")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'full': df
        }
    
    def get_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Get a small sample of data for testing purposes.
        
        Args:
            n_samples: Number of samples to return
            
        Returns:
            Sample dataframe
        """
        try:
            data = self.load_processed_data('train')
            train_df = data['train']
            return train_df.sample(n=min(n_samples, len(train_df)), random_state=42)
        except FileNotFoundError:
            logger.warning("No cached data found, loading raw data for sampling...")
            df = self.load_raw_data()
            df = self.validate_data(df)
            return df.sample(n=min(n_samples, len(df)), random_state=42)


def main():
    """Example usage of DataLoader."""
    loader = DataLoader()
    
    # Check if processed data exists
    try:
        metadata = loader.get_metadata()
        print("Found existing processed data:")
        print(f"Total samples: {metadata['total_samples']}")
        print(f"Train samples: {metadata['train_samples']}")
        print(f"Val samples: {metadata['val_samples']}")
        print(f"Test samples: {metadata['test_samples']}")
        
        # Load data
        data = loader.load_processed_data('all')
        print("\nData loaded successfully!")
        
    except FileNotFoundError:
        print("No processed data found. Processing raw data...")
        data = loader.process_data()
        print("Data processing completed!")


if __name__ == "__main__":
    main()

