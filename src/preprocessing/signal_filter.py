import numpy as np
from typing import Optional, Tuple
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ECGFilter:
    """ECG signal filtering and preprocessing."""
    
    def __init__(self, sampling_rate: int = 500):
        self.sampling_rate = sampling_rate
        self.logger = logger
        
    def bandpass_filter(self, 
                       ecg_signal: np.ndarray,
                       lowcut: float = 0.5,
                       highcut: float = 40.0,
                       order: int = 4) -> np.ndarray:
        """Apply bandpass filter to ECG signal.
        
        Args:
            ecg_signal: Input ECG signal
            lowcut: Low cutoff frequency
            highcut: High cutoff frequency
            order: Filter order
            
        Returns:
            Filtered ECG signal
        """
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, ecg_signal, axis=0)
        
        self.logger.debug(f"Applied bandpass filter: {lowcut}-{highcut} Hz")
        return filtered_signal
        
    def notch_filter(self, 
                    ecg_signal: np.ndarray,
                    notch_freq: float = 50.0,
                    quality_factor: float = 30.0) -> np.ndarray:
        """Apply notch filter to remove power line interference.
        
        Args:
            ecg_signal: Input ECG signal
            notch_freq: Frequency to remove (50 or 60 Hz)
            quality_factor: Quality factor
            
        Returns:
            Filtered ECG signal
        """
        nyquist = 0.5 * self.sampling_rate
        freq = notch_freq / nyquist
        
        b, a = iirnotch(freq, quality_factor)
        filtered_signal = filtfilt(b, a, ecg_signal, axis=0)
        
        self.logger.debug(f"Applied notch filter at {notch_freq} Hz")
        return filtered_signal
        
    def baseline_correction(self, 
                          ecg_signal: np.ndarray,
                          method: str = "median",
                          window_size: Optional[int] = None) -> np.ndarray:
        """Remove baseline drift from ECG signal.
        
        Args:
            ecg_signal: Input ECG signal
            method: Correction method ('median', 'polynomial', 'highpass')
            window_size: Window size for median filter
            
        Returns:
            Baseline-corrected ECG signal
        """
        if method == "median":
            if window_size is None:
                window_size = int(0.2 * self.sampling_rate)  # 200ms window
            
            if ecg_signal.ndim == 1:
                baseline = signal.medfilt(ecg_signal, kernel_size=window_size)
                corrected = ecg_signal - baseline
            else:
                corrected = np.zeros_like(ecg_signal)
                for i in range(ecg_signal.shape[1]):
                    baseline = signal.medfilt(ecg_signal[:, i], kernel_size=window_size)
                    corrected[:, i] = ecg_signal[:, i] - baseline
                    
        elif method == "highpass":
            # High-pass filter to remove low-frequency drift
            nyquist = 0.5 * self.sampling_rate
            cutoff = 0.5 / nyquist
            b, a = butter(4, cutoff, btype='high')
            corrected = filtfilt(b, a, ecg_signal, axis=0)
            
        elif method == "polynomial":
            # Polynomial detrending
            if ecg_signal.ndim == 1:
                x = np.arange(len(ecg_signal))
                p = np.polyfit(x, ecg_signal, deg=3)
                baseline = np.polyval(p, x)
                corrected = ecg_signal - baseline
            else:
                corrected = np.zeros_like(ecg_signal)
                for i in range(ecg_signal.shape[1]):
                    x = np.arange(len(ecg_signal))
                    p = np.polyfit(x, ecg_signal[:, i], deg=3)
                    baseline = np.polyval(p, x)
                    corrected[:, i] = ecg_signal[:, i] - baseline
        else:
            raise ValueError(f"Unknown baseline correction method: {method}")
            
        self.logger.debug(f"Applied baseline correction: {method}")
        return corrected
        
    def remove_artifacts(self, 
                        ecg_signal: np.ndarray,
                        artifact_threshold: float = 5.0) -> np.ndarray:
        """Remove artifacts and outliers from ECG signal.
        
        Args:
            ecg_signal: Input ECG signal
            artifact_threshold: Threshold for artifact detection (std units)
            
        Returns:
            Cleaned ECG signal
        """
        if ecg_signal.ndim == 1:
            # Calculate moving statistics
            window_size = int(0.1 * self.sampling_rate)  # 100ms window
            
            # Detect artifacts using z-score
            z_scores = np.abs(signal.zscore(ecg_signal))
            artifact_mask = z_scores > artifact_threshold
            
            # Interpolate over artifacts
            cleaned_signal = ecg_signal.copy()
            if np.any(artifact_mask):
                artifact_indices = np.where(artifact_mask)[0]
                valid_indices = np.where(~artifact_mask)[0]
                
                if len(valid_indices) > 0:
                    cleaned_signal[artifact_indices] = np.interp(
                        artifact_indices, valid_indices, ecg_signal[valid_indices]
                    )
        else:
            cleaned_signal = np.zeros_like(ecg_signal)
            for i in range(ecg_signal.shape[1]):
                cleaned_signal[:, i] = self.remove_artifacts(
                    ecg_signal[:, i], artifact_threshold
                )
                
        self.logger.debug(f"Removed artifacts with threshold {artifact_threshold}")
        return cleaned_signal
        
    def preprocess_ecg(self, 
                      ecg_signal: np.ndarray,
                      apply_bandpass: bool = True,
                      apply_notch: bool = True,
                      apply_baseline: bool = True,
                      remove_artifacts: bool = True) -> np.ndarray:
        """Complete ECG preprocessing pipeline.
        
        Args:
            ecg_signal: Input ECG signal
            apply_bandpass: Apply bandpass filter
            apply_notch: Apply notch filter
            apply_baseline: Apply baseline correction
            remove_artifacts: Remove artifacts
            
        Returns:
            Preprocessed ECG signal
        """
        self.logger.info("Starting ECG preprocessing pipeline...")
        
        processed_signal = ecg_signal.copy()
        
        # Apply filters in order
        if apply_bandpass:
            processed_signal = self.bandpass_filter(processed_signal)
            
        if apply_notch:
            processed_signal = self.notch_filter(processed_signal)
            
        if apply_baseline:
            processed_signal = self.baseline_correction(processed_signal)
            
        if remove_artifacts:
            processed_signal = self.remove_artifacts(processed_signal)
            
        self.logger.info("ECG preprocessing completed")
        return processed_signal
        
    def get_signal_quality(self, ecg_signal: np.ndarray) -> dict:
        """Assess ECG signal quality.
        
        Args:
            ecg_signal: Input ECG signal
            
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {}
        
        # Signal-to-noise ratio estimation
        if ecg_signal.ndim == 1:
            # Simple SNR estimation
            signal_power = np.mean(ecg_signal ** 2)
            noise_estimate = np.var(np.diff(ecg_signal))
            snr = 10 * np.log10(signal_power / noise_estimate) if noise_estimate > 0 else float('inf')
            quality_metrics['snr_db'] = snr
            
            # Amplitude range
            quality_metrics['amplitude_range'] = np.ptp(ecg_signal)
            
            # Flatline detection
            flatline_threshold = 0.01 * np.std(ecg_signal)
            flatline_samples = np.sum(np.abs(np.diff(ecg_signal)) < flatline_threshold)
            quality_metrics['flatline_percentage'] = flatline_samples / len(ecg_signal)
            
            # Saturation detection
            saturation_threshold = 0.95 * np.max(np.abs(ecg_signal))
            saturated_samples = np.sum(np.abs(ecg_signal) > saturation_threshold)
            quality_metrics['saturation_percentage'] = saturated_samples / len(ecg_signal)
            
        else:
            # Multi-lead analysis
            quality_metrics['leads'] = {}
            for i in range(ecg_signal.shape[1]):
                lead_quality = self.get_signal_quality(ecg_signal[:, i])
                quality_metrics['leads'][f'lead_{i}'] = lead_quality
                
            # Overall quality score
            lead_snrs = [metrics['snr_db'] for metrics in quality_metrics['leads'].values()]
            quality_metrics['overall_snr_db'] = np.mean(lead_snrs)
            
        return quality_metrics