"""ECG-LLM: Large Language Model-Powered ECG Analysis for Cardiovascular Diagnosis

This package provides tools for converting ECG signals into textual representations
that can be processed by Large Language Models for advanced cardiovascular diagnosis.
"""

__version__ = "0.1.0"
__author__ = "Aayush Parashar"
__email__ = "aayush.parashar@example.com"

from .utils.logger import get_logger
from .utils.data_loader import ECGDataLoader
from .models.ecg_llm import ECGLLM
from .preprocessing.signal_filter import ECGFilter
from .encoding.symbolic_encoder import SymbolicEncoder
from .encoding.template_generator import TemplateGenerator
from .evaluation.metrics import ECGMetrics

# Initialize package logger
logger = get_logger(__name__)

__all__ = [
    "ECGDataLoader",
    "ECGLLM", 
    "ECGFilter",
    "SymbolicEncoder",
    "TemplateGenerator",
    "ECGMetrics",
    "get_logger",
]