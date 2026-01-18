"""Base collector class for data collection modules."""

import logging
import requests
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from urllib.parse import urljoin, urlparse
import pandas as pd
from ..exceptions import DataCollectionError, ConfigurationError

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """Abstract base class for all data collectors."""
    
    def __init__(self, config: Dict[str, Any], download_dir: Optional[str] = None):
        """
        Initialize the collector.
        
        Args:
            config: Configuration dictionary
            download_dir: Directory to download files to
        """
        self.config = config
        self.download_dir = Path(download_dir) if download_dir else Path(config.get('download_dir', 'data/raw'))
        self.session = requests.Session()
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1)
        
        # Setup session headers
        self.session.headers.update({
            'User-Agent': 'MultiOmicsPipeline/0.1.0 (Research Purpose)'
        })
        
        # Create download directory
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized {self.__class__.__name__} with download directory: {self.download_dir}")
    
    @abstractmethod
    def collect(self, **kwargs) -> Dict[str, Any]:
        """
        Collect data from the source.
        
        Returns:
            Dictionary containing collected data and metadata
        """
        pass
    
    @abstractmethod
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for datasets.
        
        Args:
            query: Search query
            
        Returns:
            List of matching datasets
        """
        pass
    
    @abstractmethod
    def validate_dataset(self, dataset_id: str) -> bool:
        """
        Validate if a dataset exists and is accessible.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            True if dataset is valid
        """
        pass
    
    def download_file(self, url: str, filename: Optional[str] = None, 
                     chunk_size: int = 8192) -> Path:
        """
        Download a file with retry logic.
        
        Args:
            url: File URL
            filename: Local filename (optional)
            chunk_size: Download chunk size
            
        Returns:
            Path to downloaded file
        """
        if not filename:
            filename = urlparse(url).path.split('/')[-1]
        
        file_path = self.download_dir / filename
        
        # Check if file already exists
        if file_path.exists():
            logger.info(f"File already exists: {file_path}")
            return file_path
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Downloading {url} (attempt {attempt + 1}/{self.max_retries})")
                
                response = self.session.get(url, stream=True, timeout=300)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(file_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                if downloaded % (chunk_size * 100) == 0:  # Log every 100 chunks
                                    logger.debug(f"Download progress: {progress:.1f}%")
                
                logger.info(f"Successfully downloaded: {file_path}")
                return file_path
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise DataCollectionError(f"Failed to download {url} after {self.max_retries} attempts: {e}")
    
    def safe_api_call(self, url: str, method: str = 'GET', 
                     max_retries: Optional[int] = None, **kwargs) -> requests.Response:
        """
        Make a safe API call with retry logic and rate limiting.
        
        Args:
            url: API endpoint
            method: HTTP method
            max_retries: Maximum number of retries
            **kwargs: Additional arguments for requests
            
        Returns:
            API response
        """
        if max_retries is None:
            max_retries = self.max_retries
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"API call {method} {url} (attempt {attempt + 1}/{max_retries})")
                
                response = self.session.request(method, url, timeout=60, **kwargs)
                response.raise_for_status()
                
                # Rate limiting
                if 'X-RateLimit-Remaining' in response.headers:
                    remaining = int(response.headers['X-RateLimit-Remaining'])
                    if remaining < 5:
                        reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
                        wait_time = max(0, reset_time - int(time.time()))
                        logger.warning(f"Rate limit approaching. Waiting {wait_time} seconds...")
                        time.sleep(wait_time + 1)
                
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    # Check for specific error codes
                    if hasattr(e, 'response') and e.response is not None:
                        if e.response.status_code == 429:  # Rate limit
                            retry_after = int(e.response.headers.get('Retry-After', 60))
                            logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                            time.sleep(retry_after)
                        elif e.response.status_code >= 500:  # Server error
                            wait_time = self.retry_delay * (2 ** attempt)
                            logger.warning(f"Server error. Waiting {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            # Client error, don't retry
                            raise DataCollectionError(f"API call failed: {e}")
                    else:
                        # Network error, retry with backoff
                        wait_time = self.retry_delay * (2 ** attempt)
                        time.sleep(wait_time)
                else:
                    raise DataCollectionError(f"API call failed after {max_retries} attempts: {e}")
        
        raise DataCollectionError(f"API call failed after {max_retries} attempts")
    
    def validate_dataframe(self, df: pd.DataFrame, required_columns: List[str], 
                          dataset_id: str) -> bool:
        """
        Validate a dataframe.
        
        Args:
            df: DataFrame to validate
            required_columns: Required columns
            dataset_id: Dataset identifier
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If dataframe is invalid
        """
        if df.empty:
            raise DataCollectionError(f"Empty dataframe for dataset {dataset_id}")
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise DataCollectionError(
                f"Missing required columns for dataset {dataset_id}: {missing_cols}"
            )
        
        # Check for required data types (basic validation)
        if df.isnull().all().any():
            logger.warning(f"Dataset {dataset_id} has columns with all null values")
        
        return True
    
    def save_metadata(self, metadata: Dict[str, Any], filename: str) -> None:
        """
        Save dataset metadata.
        
        Args:
            metadata: Metadata dictionary
            filename: Metadata filename
        """
        import json
        from datetime import datetime
        
        metadata_file = self.download_dir / f"{filename}_metadata.json"
        
        # Add collection timestamp
        metadata['collection_timestamp'] = datetime.now().isoformat()
        metadata['collector'] = self.__class__.__name__
        
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Saved metadata to: {metadata_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")
    
    def load_metadata(self, filename: str) -> Dict[str, Any]:
        """
        Load dataset metadata.
        
        Args:
            filename: Metadata filename
            
        Returns:
            Metadata dictionary
        """
        import json
        
        metadata_file = self.download_dir / f"{filename}_metadata.json"
        
        if not metadata_file.exists():
            return {}
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            return {}
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'session'):
            self.session.close()
        logger.info(f"Cleaned up {self.__class__.__name__}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()