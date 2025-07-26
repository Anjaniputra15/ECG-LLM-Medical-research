import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ECGMetrics:
    """Comprehensive metrics for ECG diagnosis evaluation."""
    
    def __init__(self):
        self.logger = logger
        
    def calculate_classification_metrics(self, 
                                       y_true: List[str], 
                                       y_pred: List[str],
                                       average: str = 'weighted') -> Dict[str, float]:
        """Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging method for metrics
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics
        unique_labels = list(set(y_true + y_pred))
        per_class_metrics = {}
        
        for label in unique_labels:
            label_metrics = {
                'precision': precision_score(y_true, y_pred, labels=[label], average=None, zero_division=0)[0],
                'recall': recall_score(y_true, y_pred, labels=[label], average=None, zero_division=0)[0],
                'f1_score': f1_score(y_true, y_pred, labels=[label], average=None, zero_division=0)[0]
            }
            per_class_metrics[label] = label_metrics
            
        metrics['per_class'] = per_class_metrics
        
        return metrics
        
    def calculate_confidence_metrics(self, 
                                   y_true: List[str], 
                                   y_pred: List[str],
                                   confidences: List[float]) -> Dict[str, float]:
        """Calculate confidence-based metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            confidences: Prediction confidences
            
        Returns:
            Dictionary of confidence metrics
        """
        metrics = {}
        
        # Basic confidence statistics
        metrics['mean_confidence'] = np.mean(confidences)
        metrics['median_confidence'] = np.median(confidences)
        metrics['std_confidence'] = np.std(confidences)
        
        # Confidence calibration
        correct_predictions = [1 if t == p else 0 for t, p in zip(y_true, y_pred)]
        
        # Bin confidences and calculate accuracy in each bin
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(len(bins) - 1):
            bin_mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if np.any(bin_mask):
                bin_acc = np.mean([correct_predictions[j] for j in range(len(correct_predictions)) if bin_mask[j]])
                bin_conf = np.mean([confidences[j] for j in range(len(confidences)) if bin_mask[j]])
                bin_count = np.sum(bin_mask)
                
                bin_accuracies.append(bin_acc)
                bin_confidences.append(bin_conf)
                bin_counts.append(bin_count)
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append(bin_centers[i])
                bin_counts.append(0)
        
        # Expected Calibration Error (ECE)
        ece = 0.0
        total_samples = len(confidences)
        
        for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
            ece += (count / total_samples) * abs(acc - conf)
            
        metrics['ece'] = ece
        metrics['calibration_bins'] = {
            'bin_centers': bin_centers.tolist(),
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts
        }
        
        return metrics
        
    def calculate_clinical_metrics(self, 
                                 y_true: List[str], 
                                 y_pred: List[str],
                                 clinical_categories: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """Calculate clinical relevance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            clinical_categories: Mapping of clinical categories to conditions
            
        Returns:
            Dictionary of clinical metrics
        """
        if clinical_categories is None:
            clinical_categories = {
                'normal': ['Normal', 'Sinus rhythm'],
                'arrhythmia': ['Atrial fibrillation', 'Ventricular tachycardia', 'Bradycardia'],
                'ischemia': ['Myocardial infarction', 'Ischemia', 'ST elevation'],
                'conduction': ['AV block', 'Bundle branch block', 'Long QT']
            }
        
        metrics = {}
        
        # Calculate metrics for each clinical category
        for category, conditions in clinical_categories.items():
            # Convert to binary classification for this category
            y_true_binary = [1 if label in conditions else 0 for label in y_true]
            y_pred_binary = [1 if label in conditions else 0 for label in y_pred]
            
            if sum(y_true_binary) > 0:  # Only calculate if category exists in true labels
                category_metrics = {
                    'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                    'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                    'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
                    'true_positives': sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 1),
                    'false_positives': sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 1),
                    'false_negatives': sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 0),
                    'true_negatives': sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 0)
                }
                
                metrics[f'{category}_metrics'] = category_metrics
        
        # Critical condition detection
        critical_conditions = ['Myocardial infarction', 'Ventricular tachycardia', 'Ventricular fibrillation']
        
        critical_true = [1 if label in critical_conditions else 0 for label in y_true]
        critical_pred = [1 if label in critical_conditions else 0 for label in y_pred]
        
        if sum(critical_true) > 0:
            metrics['critical_detection'] = {
                'sensitivity': recall_score(critical_true, critical_pred, zero_division=0),
                'specificity': recall_score([1-x for x in critical_true], [1-x for x in critical_pred], zero_division=0),
                'missed_critical': sum(1 for t, p in zip(critical_true, critical_pred) if t == 1 and p == 0)
            }
        
        return metrics
        
    def calculate_efficiency_metrics(self, 
                                   processing_times: List[float],
                                   memory_usage: Optional[List[float]] = None) -> Dict[str, float]:
        """Calculate efficiency metrics.
        
        Args:
            processing_times: List of processing times per sample
            memory_usage: List of memory usage per sample (optional)
            
        Returns:
            Dictionary of efficiency metrics
        """
        metrics = {
            'mean_processing_time': np.mean(processing_times),
            'median_processing_time': np.median(processing_times),
            'std_processing_time': np.std(processing_times),
            'min_processing_time': np.min(processing_times),
            'max_processing_time': np.max(processing_times),
            'total_processing_time': np.sum(processing_times),
            'throughput': len(processing_times) / np.sum(processing_times)  # samples per second
        }
        
        if memory_usage is not None:
            metrics.update({
                'mean_memory_usage': np.mean(memory_usage),
                'peak_memory_usage': np.max(memory_usage),
                'memory_efficiency': len(processing_times) / np.sum(memory_usage)  # samples per MB
            })
        
        return metrics
        
    def generate_report(self, 
                       y_true: List[str], 
                       y_pred: List[str],
                       confidences: List[float],
                       processing_times: List[float],
                       model_name: str = "ECG-LLM") -> Dict[str, Any]:
        """Generate comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            confidences: Prediction confidences
            processing_times: Processing times
            model_name: Name of the model
            
        Returns:
            Comprehensive evaluation report
        """
        self.logger.info(f"Generating evaluation report for {model_name}...")
        
        report = {
            'model_name': model_name,
            'evaluation_summary': {
                'total_samples': len(y_true),
                'unique_true_labels': len(set(y_true)),
                'unique_pred_labels': len(set(y_pred))
            }
        }
        
        # Classification metrics
        report['classification_metrics'] = self.calculate_classification_metrics(y_true, y_pred)
        
        # Confidence metrics
        report['confidence_metrics'] = self.calculate_confidence_metrics(y_true, y_pred, confidences)
        
        # Clinical metrics
        report['clinical_metrics'] = self.calculate_clinical_metrics(y_true, y_pred)
        
        # Efficiency metrics
        report['efficiency_metrics'] = self.calculate_efficiency_metrics(processing_times)
        
        # Error analysis
        report['error_analysis'] = self._analyze_errors(y_true, y_pred, confidences)
        
        self.logger.info("Evaluation report generated successfully")
        
        return report
        
    def _analyze_errors(self, y_true: List[str], y_pred: List[str], confidences: List[float]) -> Dict[str, Any]:
        """Analyze prediction errors.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            confidences: Prediction confidences
            
        Returns:
            Error analysis results
        """
        errors = []
        
        for i, (true_label, pred_label, conf) in enumerate(zip(y_true, y_pred, confidences)):
            if true_label != pred_label:
                errors.append({
                    'sample_index': i,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': conf
                })
        
        # Group errors by type
        error_types = {}
        for error in errors:
            error_type = f"{error['true_label']} -> {error['predicted_label']}"
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(error)
        
        # High confidence errors (potentially concerning)
        high_conf_errors = [e for e in errors if e['confidence'] > 0.8]
        
        analysis = {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(y_true),
            'error_types': {k: len(v) for k, v in error_types.items()},
            'high_confidence_errors': len(high_conf_errors),
            'most_common_errors': sorted(error_types.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        }
        
        return analysis
        
    def save_report(self, report: Dict[str, Any], filepath: str):
        """Save evaluation report to file.
        
        Args:
            report: Evaluation report
            filepath: Output file path
        """
        import json
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation report saved to {filepath}")
        
    def print_summary(self, report: Dict[str, Any]):
        """Print evaluation summary.
        
        Args:
            report: Evaluation report
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION REPORT: {report['model_name']}")
        print(f"{'='*60}")
        
        # Summary
        summary = report['evaluation_summary']
        print(f"Total Samples: {summary['total_samples']}")
        print(f"Unique True Labels: {summary['unique_true_labels']}")
        print(f"Unique Predicted Labels: {summary['unique_pred_labels']}")
        
        # Classification metrics
        cls_metrics = report['classification_metrics']
        print(f"\nClassification Metrics:")
        print(f"  Accuracy: {cls_metrics['accuracy']:.3f}")
        print(f"  Precision: {cls_metrics['precision']:.3f}")
        print(f"  Recall: {cls_metrics['recall']:.3f}")
        print(f"  F1-Score: {cls_metrics['f1_score']:.3f}")
        
        # Confidence metrics
        conf_metrics = report['confidence_metrics']
        print(f"\nConfidence Metrics:")
        print(f"  Mean Confidence: {conf_metrics['mean_confidence']:.3f}")
        print(f"  Calibration Error (ECE): {conf_metrics['ece']:.3f}")
        
        # Efficiency metrics
        eff_metrics = report['efficiency_metrics']
        print(f"\nEfficiency Metrics:")
        print(f"  Mean Processing Time: {eff_metrics['mean_processing_time']:.3f}s")
        print(f"  Throughput: {eff_metrics['throughput']:.1f} samples/s")
        
        # Error analysis
        error_analysis = report['error_analysis']
        print(f"\nError Analysis:")
        print(f"  Total Errors: {error_analysis['total_errors']}")
        print(f"  Error Rate: {error_analysis['error_rate']:.3f}")
        print(f"  High Confidence Errors: {error_analysis['high_confidence_errors']}")
        
        print(f"{'='*60}\n")