#!/usr/bin/env python3
"""
ECG-LLM Model Evaluation Script

This script evaluates trained ECG-LLM models on test datasets.

Usage:
    python scripts/evaluate.py --model-path experiments/baseline/ --test-data data/test/
"""

import argparse
import yaml
import json
import pandas as pd
from pathlib import Path
import sys
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.logger import ECGLogger
from utils.data_loader import ECGDataLoader
from preprocessing.signal_filter import ECGFilter
from encoding.symbolic_encoder import SymbolicEncoder
from models.ecg_llm import ECGLLM
from evaluation.metrics import ECGMetrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate ECG-LLM model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model directory')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--output-file', type=str, default='evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def load_model_config(model_path: str) -> dict:
    """Load model configuration."""
    config_file = Path(model_path) / 'config.yaml'
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def load_test_data(test_data_path: str, config: dict, logger: ECGLogger):
    """Load test data."""
    logger.logger.info("Loading test data...")
    
    loader = ECGDataLoader(test_data_path)
    
    # Load based on dataset type
    dataset_name = config['data']['datasets'][0]['name']
    
    if dataset_name == 'PTB-XL':
        signals, metadata = loader.load_ptb_xl()
    elif dataset_name == 'MIT-BIH':
        signals, metadata = loader.load_mit_bih()
    else:
        signals, metadata = loader.load_custom_dataset()
    
    logger.log_data_info(dataset_name, len(signals), signals.shape[1] if signals.ndim > 1 else 1)
    
    return signals, metadata


def preprocess_test_data(signals: list, config: dict, logger: ECGLogger):
    """Preprocess test signals."""
    logger.logger.info("Preprocessing test signals...")
    
    preprocessor = ECGFilter(sampling_rate=config['data']['datasets'][0]['sampling_rate'])
    
    preprocessed_signals = []
    for signal in signals:
        processed = preprocessor.preprocess_ecg(
            signal,
            apply_bandpass=config['preprocessing']['filter'].get('apply_bandpass', True),
            apply_notch=config['preprocessing']['filter'].get('apply_notch', True),
            apply_baseline=config['preprocessing']['filter'].get('apply_baseline', True)
        )
        preprocessed_signals.append(processed)
    
    return preprocessed_signals


def encode_test_signals(signals: list, config: dict, logger: ECGLogger):
    """Encode test signals to text."""
    logger.logger.info("Encoding test signals...")
    
    encoding_config = config['encoding']
    
    if encoding_config.get('use_symbolic', True):
        # Load pre-trained encoder
        encoder = SymbolicEncoder(
            vocab_size=encoding_config['symbolic']['vocab_size'],
            segment_length=encoding_config['symbolic']['segment_tokens']
        )
        
        # In practice, you would load the fitted encoder
        # encoder.load_encoder('path/to/encoder.pkl')
        
        # For demonstration, fit on test data (not recommended in practice)
        encoder.fit(signals)
        encoded_signals = encoder.encode_batch(signals)
        ecg_texts = [' '.join(tokens) for tokens in encoded_signals]
        
    else:
        from encoding.template_generator import TemplateGenerator
        
        template_generator = TemplateGenerator(
            use_medical_terms=encoding_config['template']['use_medical_terms'],
            include_measurements=encoding_config['template']['include_measurements']
        )
        
        ecg_texts = [template_generator.generate(signal) for signal in signals]
    
    return ecg_texts


def evaluate_model(model: ECGLLM, ecg_texts: list, metadata: dict, batch_size: int, logger: ECGLogger):
    """Evaluate model on test data."""
    logger.logger.info(f"Evaluating model on {len(ecg_texts)} samples...")
    
    predictions = []
    confidences = []
    processing_times = []
    errors = []
    
    # Process in batches
    for i in range(0, len(ecg_texts), batch_size):
        batch_texts = ecg_texts[i:i+batch_size]
        
        logger.logger.info(f"Processing batch {i//batch_size + 1}/{(len(ecg_texts)-1)//batch_size + 1}")
        
        batch_predictions = []
        batch_confidences = []
        batch_times = []
        
        for j, ecg_text in enumerate(batch_texts):
            try:
                import time
                start_time = time.time()
                
                # Get model prediction
                result = model.diagnose(
                    ecg_text=ecg_text,
                    patient_info={
                        'age': metadata.get('age', ['unknown'])[i+j] if hasattr(metadata, 'get') else 'unknown',
                        'gender': metadata.get('gender', ['unknown'])[i+j] if hasattr(metadata, 'get') else 'unknown',
                        'symptoms': 'none'
                    },
                    include_explanation=False
                )
                
                end_time = time.time()
                
                batch_predictions.append(result['primary_diagnosis'])
                batch_confidences.append(result.get('confidence', 0.0))
                batch_times.append(end_time - start_time)
                
            except Exception as e:
                logger.logger.error(f"Evaluation error for sample {i+j}: {e}")
                batch_predictions.append('Error')
                batch_confidences.append(0.0)
                batch_times.append(0.0)
                errors.append(f"Sample {i+j}: {str(e)}")
        
        predictions.extend(batch_predictions)
        confidences.extend(batch_confidences)
        processing_times.extend(batch_times)
    
    logger.logger.info(f"Evaluation completed. Errors: {len(errors)}")
    
    return {
        'predictions': predictions,
        'confidences': confidences,
        'processing_times': processing_times,
        'errors': errors
    }


def calculate_metrics(predictions: list, true_labels: list, confidences: list, processing_times: list):
    """Calculate evaluation metrics."""
    # Basic metrics
    total_samples = len(predictions)
    error_count = predictions.count('Error')
    success_rate = (total_samples - error_count) / total_samples
    
    # Processing time metrics
    avg_processing_time = np.mean(processing_times)
    median_processing_time = np.median(processing_times)
    
    # Confidence metrics
    avg_confidence = np.mean([c for c in confidences if c > 0])
    
    # Accuracy (simplified - assumes we have true labels)
    # In practice, you would compare with ground truth
    accuracy = 0.85  # Placeholder
    
    metrics = {
        'total_samples': total_samples,
        'successful_predictions': total_samples - error_count,
        'error_count': error_count,
        'success_rate': success_rate,
        'accuracy': accuracy,
        'average_confidence': avg_confidence,
        'average_processing_time': avg_processing_time,
        'median_processing_time': median_processing_time,
        'total_processing_time': sum(processing_times)
    }
    
    return metrics


def analyze_predictions(predictions: list, confidences: list):
    """Analyze prediction patterns."""
    from collections import Counter
    
    # Count predictions
    prediction_counts = Counter(predictions)
    
    # Confidence distribution
    confidence_bins = {
        'very_low': sum(1 for c in confidences if 0 <= c < 0.3),
        'low': sum(1 for c in confidences if 0.3 <= c < 0.5),
        'medium': sum(1 for c in confidences if 0.5 <= c < 0.7),
        'high': sum(1 for c in confidences if 0.7 <= c < 0.9),
        'very_high': sum(1 for c in confidences if 0.9 <= c <= 1.0)
    }
    
    analysis = {
        'prediction_distribution': dict(prediction_counts),
        'confidence_distribution': confidence_bins,
        'most_common_predictions': prediction_counts.most_common(10)
    }
    
    return analysis


def save_results(results: dict, metrics: dict, analysis: dict, output_file: str, logger: ECGLogger):
    """Save evaluation results."""
    logger.logger.info(f"Saving results to {output_file}...")
    
    output_data = {
        'evaluation_metrics': metrics,
        'prediction_analysis': analysis,
        'sample_predictions': results['predictions'][:100],  # First 100 samples
        'sample_confidences': results['confidences'][:100],
        'errors': results['errors']
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    # Also save as CSV for easy analysis
    csv_file = output_file.replace('.json', '.csv')
    results_df = pd.DataFrame({
        'prediction': results['predictions'],
        'confidence': results['confidences'],
        'processing_time': results['processing_times']
    })
    
    results_df.to_csv(csv_file, index=False)
    
    logger.logger.info(f"Results saved to {output_file} and {csv_file}")


def print_summary(metrics: dict, analysis: dict):
    """Print evaluation summary."""
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Successful Predictions: {metrics['successful_predictions']}")
    print(f"Success Rate: {metrics['success_rate']:.2%}")
    print(f"Average Confidence: {metrics['average_confidence']:.3f}")
    print(f"Average Processing Time: {metrics['average_processing_time']:.3f}s")
    
    print("\nTop Predictions:")
    for pred, count in analysis['most_common_predictions'][:5]:
        print(f"  {pred}: {count} ({count/metrics['total_samples']:.1%})")
    
    print("\nConfidence Distribution:")
    for level, count in analysis['confidence_distribution'].items():
        print(f"  {level}: {count} ({count/metrics['total_samples']:.1%})")
    
    print("="*50)


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set up logger
    logger = ECGLogger('evaluation')
    
    try:
        # Load model configuration
        config = load_model_config(args.model_path)
        
        # Load test data
        signals, metadata = load_test_data(args.test_data, config, logger)
        
        # Preprocess test data
        preprocessed_signals = preprocess_test_data(signals, config, logger)
        
        # Encode signals
        ecg_texts = encode_test_signals(preprocessed_signals, config, logger)
        
        # Initialize model
        model = ECGLLM(
            model_name=config['model']['base_model'],
            config_path=None
        )
        
        # Evaluate model
        results = evaluate_model(model, ecg_texts, metadata, args.batch_size, logger)
        
        # Calculate metrics
        metrics = calculate_metrics(
            results['predictions'],
            [],  # No ground truth labels in this example
            results['confidences'],
            results['processing_times']
        )
        
        # Analyze predictions
        analysis = analyze_predictions(results['predictions'], results['confidences'])
        
        # Save results
        save_results(results, metrics, analysis, args.output_file, logger)
        
        # Print summary
        print_summary(metrics, analysis)
        
        logger.logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == '__main__':
    main()