"""Performance monitoring and optimization utilities for multi-omics pipeline."""

import time
import psutil
import os
import gc
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from functools import wraps
import threading
import json

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize performance monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.enable_monitoring = self.config.get('enable_monitoring', True)
        self.track_memory = self.config.get('track_memory', True)
        self.track_cpu = self.config.get('track_cpu', True)
        self.track_disk = self.config.get('track_disk', True)
        
        self.metrics = []
        self.start_time = None
        self.monitoring = False
        self.monitor_thread = None
        
        logger.info("Initialized PerformanceMonitor")
    
    def start_monitoring(self, task_name: str = "pipeline_execution"):
        """Start performance monitoring."""
        if not self.enable_monitoring:
            return
        
        self.start_time = time.time()
        self.monitoring = True
        self.metrics = []
        
        # Start background monitoring thread
        if self.track_memory or self.track_cpu:
            self.monitor_thread = threading.Thread(
                target=self._background_monitoring,
                args=(task_name,),
                daemon=True
            )
            self.monitor_thread.start()
        
        logger.info(f"Performance monitoring started for: {task_name}")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop performance monitoring and return summary."""
        if not self.enable_monitoring or not self.monitoring:
            return {}
        
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        total_time = time.time() - self.start_time
        
        # Calculate summary statistics
        summary = self._calculate_summary_metrics()
        summary['total_execution_time'] = total_time
        summary['monitoring_duration'] = total_time
        
        logger.info("Performance monitoring stopped")
        logger.info(f"Execution summary: {summary}")
        
        return summary
    
    def _background_monitoring(self, task_name: str):
        """Background monitoring of system resources."""
        while self.monitoring:
            try:
                metric = {
                    'timestamp': datetime.now().isoformat(),
                    'task_name': task_name,
                    'elapsed_time': time.time() - self.start_time
                }
                
                # Memory metrics
                if self.track_memory:
                    process = psutil.Process(os.getpid())
                    metric.update({
                        'memory_mb': process.memory_info().rss / (1024 * 1024),
                        'memory_percent': process.memory_percent(),
                        'available_memory_gb': psutil.virtual_memory().available / (1024**3),
                        'memory_usage_percent': psutil.virtual_memory().percent
                    })
                
                # CPU metrics
                if self.track_cpu:
                    metric.update({
                        'cpu_percent': psutil.cpu_percent(interval=1),
                        'cpu_count': psutil.cpu_count(),
                        'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else None
                    })
                
                # Disk metrics
                if self.track_disk:
                    disk_usage = psutil.disk_usage('/')
                    metric.update({
                        'disk_usage_percent': (disk_usage.used / disk_usage.total) * 100,
                        'disk_free_gb': disk_usage.free / (1024**3)
                    })
                
                self.metrics.append(metric)
                
                # Sleep for monitoring interval
                time.sleep(self.config.get('monitoring_interval', 5))
                
            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _calculate_summary_metrics(self) -> Dict[str, Any]:
        """Calculate summary statistics from collected metrics."""
        if not self.metrics:
            return {}
        
        metrics_df = pd.DataFrame(self.metrics)
        
        summary = {
            'peak_memory_mb': metrics_df['memory_mb'].max() if 'memory_mb' in metrics_df else None,
            'avg_memory_mb': metrics_df['memory_mb'].mean() if 'memory_mb' in metrics_df else None,
            'peak_cpu_percent': metrics_df['cpu_percent'].max() if 'cpu_percent' in metrics_df else None,
            'avg_cpu_percent': metrics_df['cpu_percent'].mean() if 'cpu_percent' in metrics_df else None,
            'peak_disk_usage_percent': metrics_df['disk_usage_percent'].max() if 'disk_usage_percent' in metrics_df else None,
            'n_measurements': len(metrics_df),
            'resource_efficiency_score': self._calculate_efficiency_score(metrics_df)
        }
        
        return summary
    
    def _calculate_efficiency_score(self, metrics_df: pd.DataFrame) -> float:
        """Calculate resource efficiency score (0-100)."""
        score = 100.0
        
        # Memory efficiency (lower is better)
        if 'memory_percent' in metrics_df:
            avg_memory = metrics_df['memory_percent'].mean()
            if avg_memory > 80:
                score -= 20
            elif avg_memory > 60:
                score -= 10
        
        # CPU efficiency (lower is better for background tasks)
        if 'cpu_percent' in metrics_df:
            avg_cpu = metrics_df['cpu_percent'].mean()
            if avg_cpu > 80:
                score -= 15
            elif avg_cpu > 60:
                score -= 7
        
        # Disk space efficiency
        if 'disk_usage_percent' in metrics_df:
            avg_disk = metrics_df['disk_usage_percent'].mean()
            if avg_disk > 90:
                score -= 15
            elif avg_disk > 80:
                score -= 7
        
        return max(0, score)
    
    def save_metrics(self, output_path: str):
        """Save metrics to file."""
        if not self.metrics:
            logger.warning("No metrics to save")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        metrics_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'n_measurements': len(self.metrics),
                'config': self.config
            },
            'metrics': self.metrics,
            'summary': self._calculate_summary_metrics()
        }
        
        with open(output_path, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        logger.info(f"Metrics saved to: {output_path}")


def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get performance monitor if available
        monitor = kwargs.get('performance_monitor', PerformanceMonitor())
        
        if monitor.enable_monitoring:
            task_name = f"{func.__module__}.{func.__name__}"
            monitor.start_monitoring(task_name)
        
        try:
            result = func(*args, **kwargs)
            
            if monitor.enable_monitoring:
                summary = monitor.stop_monitoring()
                logger.info(f"Function {func.__name__} completed with summary: {summary}")
            
            return result
            
        except Exception as e:
            if monitor.enable_monitoring:
                monitor.stop_monitoring()
            raise e
    
    return wrapper


class MemoryOptimizer:
    """Optimize memory usage for large datasets."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize memory optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.aggressive_optimization = self.config.get('aggressive_optimization', False)
        self.chunk_size = self.config.get('chunk_size', 10000)
        
        logger.info("Initialized MemoryOptimizer")
    
    def optimize_dataframe(self, df: pd.DataFrame, 
                          category_threshold: float = 0.5) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.
        
        Args:
            df: Input DataFrame
            category_threshold: Threshold for converting to category type
            
        Returns:
            Optimized DataFrame
        """
        initial_memory = df.memory_usage(deep=True).sum()
        logger.info(f"Starting DataFrame optimization. Initial memory: {initial_memory / (1024**2):.2f} MB")
        
        df_optimized = df.copy()
        
        # Optimize numeric columns
        for col in df_optimized.select_dtypes(include=['int64', 'int32']).columns:
            col_min = df_optimized[col].min()
            col_max = df_optimized[col].max()
            
            if col_min >= 0:  # Unsigned integers
                if col_max < 255:
                    df_optimized[col] = df_optimized[col].astype(np.uint8)
                elif col_max < 65535:
                    df_optimized[col] = df_optimized[col].astype(np.uint16)
                elif col_max < 4294967295:
                    df_optimized[col] = df_optimized[col].astype(np.uint32)
            else:  # Signed integers
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df_optimized[col] = df_optimized[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df_optimized[col] = df_optimized[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df_optimized[col] = df_optimized[col].astype(np.int32)
        
        # Optimize float columns
        for col in df_optimized.select_dtypes(include=['float64']).columns:
            df_optimized[col] = df_optimized[col].astype(np.float32)
        
        # Optimize object columns
        for col in df_optimized.select_dtypes(include=['object']).columns:
            num_unique_values = len(df_optimized[col].unique())
            num_total_values = len(df_optimized[col])
            
            if num_unique_values / num_total_values < category_threshold:
                df_optimized[col] = df_optimized[col].astype('category')
        
        # Final memory usage
        final_memory = df_optimized.memory_usage(deep=True).sum()
        memory_reduction = (initial_memory - final_memory) / initial_memory * 100
        
        logger.info(f"DataFrame optimization complete. Final memory: {final_memory / (1024**2):.2f} MB "
                   f"({memory_reduction:.1f}% reduction)")
        
        return df_optimized
    
    def chunk_process_dataframe(self, df: pd.DataFrame, 
                              process_func: Callable[[pd.DataFrame], pd.DataFrame],
                              chunk_size: Optional[int] = None) -> pd.DataFrame:
        """
        Process DataFrame in chunks to manage memory.
        
        Args:
            df: Input DataFrame
            process_func: Function to apply to each chunk
            chunk_size: Size of each chunk
            
        Returns:
            Processed DataFrame
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        n_chunks = (len(df) + chunk_size - 1) // chunk_size
        logger.info(f"Processing DataFrame in {n_chunks} chunks of size {chunk_size}")
        
        processed_chunks = []
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            
            # Process chunk
            processed_chunk = process_func(chunk)
            processed_chunks.append(processed_chunk)
            
            # Clear memory periodically
            if i % (chunk_size * 10) == 0:
                gc.collect()
                logger.debug(f"Processed chunk {i // chunk_size + 1}/{n_chunks}")
        
        # Combine chunks
        result = pd.concat(processed_chunks, ignore_index=True)
        
        # Final cleanup
        gc.collect()
        
        logger.info(f"Chunk processing complete. Result shape: {result.shape}")
        
        return result
    
    def clear_memory(self):
        """Force garbage collection and clear memory."""
        initial_memory = psutil.virtual_memory().percent
        
        # Clear DataFrame cache
        pd.core.common.clear_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Clear matplotlib cache if available
        try:
            plt.close('all')
        except:
            pass
        
        final_memory = psutil.virtual_memory().percent
        
        logger.info(f"Memory cleared. Usage: {initial_memory:.1f}% -> {final_memory:.1f}%")


class ParallelProcessor:
    """Handle parallel processing with performance monitoring."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize parallel processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.max_workers = self.config.get('max_workers', min(4, os.cpu_count()))
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.timeout = self.config.get('timeout', 300)  # 5 minutes
        
        logger.info(f"Initialized ParallelProcessor with {self.max_workers} workers")
    
    def parallel_map(self, func: Callable, data: List[Any], 
                    *args, **kwargs) -> List[Any]:
        """
        Apply function to data in parallel.
        
        Args:
            func: Function to apply
            data: List of data items
            *args: Additional arguments for func
            **kwargs: Additional keyword arguments
            
        Returns:
            List of results
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        logger.info(f"Starting parallel processing of {len(data)} items with {self.max_workers} workers")
        
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            futures = []
            for i, item in enumerate(data):
                future = executor.submit(func, item, *args, **kwargs)
                futures.append((i, future))
            
            # Collect results
            for i, future in as_completed(futures, timeout=self.timeout):
                try:
                    result = future.result(timeout=self.timeout)
                    results.append((i, result))
                except Exception as e:
                    logger.error(f"Error processing item {i}: {e}")
                    results.append((i, None))
        
        # Sort by original order
        results.sort(key=lambda x: x[0])
        
        logger.info(f"Parallel processing completed. Success rate: {sum(1 for _, r in results if r is not None)}/{len(results)}")
        
        return [result for _, result in results]


class CachingOptimizer:
    """Implement caching for expensive operations."""
    
    def __init__(self, cache_dir: str = "cache", max_cache_size: int = 1000):
        """
        Initialize caching optimizer.
        
        Args:
            cache_dir: Directory for cache files
            max_cache_size: Maximum number of cached items
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        self.cache_index = {}
        
        logger.info(f"Initialized CachingOptimizer with cache directory: {cache_dir}")
    
    def cache_result(self, func_name: str, key: str, result: Any):
        """Cache function result."""
        cache_file = self.cache_dir / f"{func_name}_{hash(key)}.pkl"
        
        try:
            import joblib
            joblib.dump(result, cache_file)
            
            # Update cache index
            self.cache_index[key] = {
                'file': str(cache_file),
                'timestamp': datetime.now().isoformat(),
                'func_name': func_name
            }
            
            # Clean old cache if needed
            self._cleanup_cache()
            
            logger.debug(f"Cached result for {func_name} with key: {key}")
            
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    def get_cached_result(self, func_name: str, key: str) -> Optional[Any]:
        """Get cached result if available."""
        if key not in self.cache_index:
            return None
        
        try:
            import joblib
            
            cache_info = self.cache_index[key]
            cache_file = Path(cache_info['file'])
            
            if cache_file.exists():
                result = joblib.load(cache_file)
                logger.debug(f"Retrieved cached result for {func_name} with key: {key}")
                return result
            else:
                # Remove from index if file doesn't exist
                del self.cache_index[key]
                return None
                
        except Exception as e:
            logger.warning(f"Failed to retrieve cached result: {e}")
            return None
    
    def _cleanup_cache(self):
        """Remove old cache files if over limit."""
        if len(self.cache_index) <= self.max_cache_size:
            return
        
        # Sort by timestamp and remove oldest
        sorted_items = sorted(self.cache_index.items(), 
                            key=lambda x: x[1]['timestamp'])
        
        # Remove oldest items
        items_to_remove = sorted_items[:len(self.cache_index) - self.max_cache_size + 1]
        
        for key, cache_info in items_to_remove:
            try:
                cache_file = Path(cache_info['file'])
                if cache_file.exists():
                    cache_file.unlink()
                del self.cache_index[key]
            except Exception as e:
                logger.warning(f"Failed to remove cache file: {e}")


def cache_result(cache_key_func: Callable[[tuple, dict], str]):
    """Decorator to cache function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = cache_key_func(args, kwargs)
            
            # Get cache optimizer (assuming it's passed or global)
            cache_optimizer = kwargs.get('cache_optimizer', CachingOptimizer())
            
            # Check cache
            cached_result = cache_optimizer.get_cached_result(func.__name__, cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache_optimizer.cache_result(func.__name__, cache_key, result)
            
            return result
        
        return wrapper
    return decorator