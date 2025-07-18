import logging
import os
from datetime import datetime
from typing import Optional


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Create and configure a logger.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # File handler
        file_handler = logging.FileHandler("logs/ecg_llm.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Set level
        logger.setLevel(getattr(logging, level.upper()))
    
    return logger


class ECGLogger:
    """Enhanced logger for ECG analysis workflows."""
    
    def __init__(self, name: str, experiment_name: Optional[str] = None):
        self.logger = get_logger(name)
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        
    def log_experiment_start(self, config: dict):
        """Log experiment start with configuration."""
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        self.logger.info(f"Configuration: {config}")
        
    def log_data_info(self, dataset_name: str, n_samples: int, n_features: int):
        """Log dataset information."""
        self.logger.info(
            f"Dataset: {dataset_name}, Samples: {n_samples}, Features: {n_features}"
        )
        
    def log_model_info(self, model_name: str, n_parameters: int):
        """Log model information."""
        self.logger.info(
            f"Model: {model_name}, Parameters: {n_parameters:,}"
        )
        
    def log_metrics(self, metrics: dict, epoch: Optional[int] = None):
        """Log evaluation metrics."""
        prefix = f"Epoch {epoch} - " if epoch is not None else ""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"{prefix}Metrics: {metrics_str}")
        
    def log_experiment_end(self):
        """Log experiment completion."""
        duration = datetime.now() - self.start_time
        self.logger.info(
            f"Experiment completed in {duration.total_seconds():.2f} seconds"
        )