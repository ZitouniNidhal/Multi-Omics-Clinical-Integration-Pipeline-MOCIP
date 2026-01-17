"""Logging utilities for the multi-omics pipeline."""

import logging
import logging.config
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import sys
import json
from pythonjsonlogger import jsonlogger


def setup_logging(config_path: Optional[str] = None, 
                  log_level: str = 'INFO',
                  log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        config_path: Path to logging configuration file
        log_level: Logging level
        log_file: Path to log file
        
    Returns:
        Configured logger
    """
    if config_path and Path(config_path).exists():
        # Load configuration from file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        # Use default configuration
        default_config = get_default_logging_config(log_level, log_file)
        logging.config.dictConfig(default_config)
    
    # Get logger
    logger = logging.getLogger('multiomics_pipeline')
    logger.info("Logging setup complete")
    
    return logger


def get_default_logging_config(log_level: str = 'INFO', 
                              log_file: Optional[str] = None) -> Dict[str, Any]:
    """Get default logging configuration."""
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'json': {
                'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
                'format': '%(asctime)s %(name)s %(levelname)s %(funcName)s %(lineno)d %(message)s'
            },
            'simple': {
                'format': '%(asctime)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            }
        },
        'loggers': {
            'multiomics_pipeline': {
                'level': log_level,
                'handlers': ['console'],
                'propagate': False
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console']
        }
    }
    
    # Add file handler if log file specified
    if log_file:
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': log_level,
            'formatter': 'standard',
            'filename': log_file,
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'encoding': 'utf8'
        }
        
        # Add file handler to loggers
        config['loggers']['multiomics_pipeline']['handlers'].append('file')
        config['root']['handlers'].append('file')
    
    return config


class PipelineLogger:
    """Custom logger for the multi-omics pipeline with additional functionality."""
    
    def __init__(self, name: str, log_level: str = 'INFO'):
        """
        Initialize pipeline logger.
        
        Args:
            name: Logger name
            log_level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Add custom handlers if needed
        self._setup_custom_handlers()
        
        self.pipeline_start_time = None
        self.step_times = {}
    
    def _setup_custom_handlers(self):
        """Setup custom logging handlers."""
        # JSON formatter for structured logging
        json_handler = logging.StreamHandler()
        json_handler.setFormatter(
            jsonlogger.JsonFormatter(
                '%(asctime)s %(name)s %(levelname)s %(funcName)s %(lineno)d %(message)s'
            )
        )
        
        # Add handler if not already present
        if not any(isinstance(h, type(json_handler)) for h in self.logger.handlers):
            self.logger.addHandler(json_handler)
    
    def start_pipeline(self, pipeline_name: str, config: Dict[str, Any]):
        """Log pipeline start with configuration."""
        self.pipeline_start_time = datetime.now()
        
        self.logger.info("Pipeline started", extra={
            'event': 'pipeline_start',
            'pipeline_name': pipeline_name,
            'config': config,
            'timestamp': self.pipeline_start_time.isoformat()
        })
    
    def end_pipeline(self, pipeline_name: str, status: str = 'success', error: Optional[str] = None):
        """Log pipeline end with summary."""
        if self.pipeline_start_time:
            duration = (datetime.now() - self.pipeline_start_time).total_seconds()
            
            extra_data = {
                'event': 'pipeline_end',
                'pipeline_name': pipeline_name,
                'status': status,
                'duration_seconds': duration,
                'timestamp': datetime.now().isoformat()
            }
            
            if error:
                extra_data['error'] = error
            
            self.logger.info("Pipeline completed", extra=extra_data)
        else:
            self.logger.warning("Pipeline end called without start time")
    
    def start_step(self, step_name: str, **kwargs):
        """Log step start."""
        self.step_times[step_name] = datetime.now()
        
        self.logger.info(f"Step started: {step_name}", extra={
            'event': 'step_start',
            'step_name': step_name,
            'timestamp': self.step_times[step_name].isoformat(),
            'kwargs': kwargs
        })
    
    def end_step(self, step_name: str, status: str = 'success', **kwargs):
        """Log step end with timing."""
        if step_name in self.step_times:
            start_time = self.step_times[step_name]
            duration = (datetime.now() - start_time).total_seconds()
            
            extra_data = {
                'event': 'step_end',
                'step_name': step_name,
                'status': status,
                'duration_seconds': duration,
                'timestamp': datetime.now().isoformat()
            }
            extra_data.update(kwargs)
            
            self.logger.info(f"Step completed: {step_name}", extra=extra_data)
            
            # Remove from tracking
            del self.step_times[step_name]
        else:
            self.logger.warning(f"Step end called without start time: {step_name}")
    
    def log_data_info(self, data_name: str, data_info: Dict[str, Any]):
        """Log data information."""
        self.logger.info(f"Data info: {data_name}", extra={
            'event': 'data_info',
            'data_name': data_name,
            'data_info': data_info
        })
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics."""
        self.logger.info("Performance metrics", extra={
            'event': 'performance_metrics',
            'metrics': metrics
        })
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log error with detailed context."""
        self.logger.error(f"Error occurred: {str(error)}", extra={
            'event': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat()
        })


def get_logger(name: str = 'multiomics_pipeline') -> PipelineLogger:
    """Get configured logger instance."""
    return PipelineLogger(name)


# Convenience functions
def log_step(func):
    """Decorator to automatically log function execution."""
    def wrapper(*args, **kwargs):
        logger = get_logger()
        step_name = func.__name__
        
        logger.start_step(step_name)
        try:
            result = func(*args, **kwargs)
            logger.end_step(step_name, status='success')
            return result
        except Exception as e:
            logger.end_step(step_name, status='failed', error=str(e))
            raise
    
    return wrapper


def log_pipeline_execution(pipeline_func):
    """Decorator to log entire pipeline execution."""
    def wrapper(*args, **kwargs):
        logger = get_logger()
        pipeline_name = pipeline_func.__name__
        
        try:
            # Extract config if available
            config = kwargs.get('config', {})
            
            logger.start_pipeline(pipeline_name, config)
            result = pipeline_func(*args, **kwargs)
            logger.end_pipeline(pipeline_name, status='success')
            
            return result
            
        except Exception as e:
            logger.end_pipeline(pipeline_name, status='failed', error=str(e))
            raise
    
    return wrapper