import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import wfdb
from .logger import get_logger

logger = get_logger(__name__)


class ECGDataLoader:
    """Universal ECG data loader for multiple formats and datasets."""
    
    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)
        self.logger = logger
        
    def load_ptb_xl(self, sampling_rate: int = 500) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load PTB-XL dataset.
        
        Args:
            sampling_rate: Target sampling rate
            
        Returns:
            Tuple of (signals, metadata)
        """
        self.logger.info("Loading PTB-XL dataset...")
        
        # Load metadata
        metadata_path = self.data_path / "ptbxl_database.csv"
        metadata = pd.read_csv(metadata_path)
        
        # Load signal data
        signals = []
        for idx, row in metadata.iterrows():
            signal_path = self.data_path / row['filename_hr']
            signal = self._load_wfdb_signal(signal_path, sampling_rate)
            signals.append(signal)
            
        signals = np.array(signals)
        self.logger.info(f"Loaded {len(signals)} PTB-XL records")
        
        return signals, metadata
        
    def load_mit_bih(self, sampling_rate: int = 360) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load MIT-BIH dataset.
        
        Args:
            sampling_rate: Target sampling rate
            
        Returns:
            Tuple of (signals, metadata)
        """
        self.logger.info("Loading MIT-BIH dataset...")
        
        # Find all .dat files
        dat_files = list(self.data_path.glob("*.dat"))
        
        signals = []
        metadata = []
        
        for dat_file in dat_files:
            record_name = dat_file.stem
            try:
                signal = self._load_wfdb_signal(dat_file, sampling_rate)
                signals.append(signal)
                metadata.append({"record_name": record_name, "filename": str(dat_file)})
            except Exception as e:
                self.logger.warning(f"Failed to load {record_name}: {e}")
                
        signals = np.array(signals)
        metadata = pd.DataFrame(metadata)
        
        self.logger.info(f"Loaded {len(signals)} MIT-BIH records")
        
        return signals, metadata
        
    def load_custom_dataset(self, format_type: str = "csv") -> Tuple[np.ndarray, pd.DataFrame]:
        """Load custom dataset.
        
        Args:
            format_type: Format type ('csv', 'npy', 'mat')
            
        Returns:
            Tuple of (signals, metadata)
        """
        self.logger.info(f"Loading custom dataset in {format_type} format...")
        
        if format_type == "csv":
            return self._load_csv_dataset()
        elif format_type == "npy":
            return self._load_npy_dataset()
        elif format_type == "mat":
            return self._load_mat_dataset()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
            
    def _load_wfdb_signal(self, signal_path: Path, sampling_rate: int) -> np.ndarray:
        """Load WFDB format signal."""
        record = wfdb.rdrecord(str(signal_path.with_suffix('')))
        signal = record.p_signal
        
        # Resample if needed
        if record.fs != sampling_rate:
            signal = self._resample_signal(signal, record.fs, sampling_rate)
            
        return signal
        
    def _load_csv_dataset(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load CSV format dataset."""
        csv_files = list(self.data_path.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found in data path")
            
        # Load first file to get structure
        sample_df = pd.read_csv(csv_files[0])
        
        # Assume signal columns are numeric and metadata columns are non-numeric
        signal_columns = sample_df.select_dtypes(include=[np.number]).columns
        metadata_columns = sample_df.select_dtypes(exclude=[np.number]).columns
        
        all_signals = []
        all_metadata = []
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            signals = df[signal_columns].values
            metadata = df[metadata_columns].iloc[0].to_dict()
            metadata["filename"] = str(csv_file)
            
            all_signals.append(signals)
            all_metadata.append(metadata)
            
        return np.array(all_signals), pd.DataFrame(all_metadata)
        
    def _load_npy_dataset(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load NumPy format dataset."""
        npy_files = list(self.data_path.glob("*.npy"))
        
        if not npy_files:
            raise FileNotFoundError("No NPY files found in data path")
            
        signals = []
        metadata = []
        
        for npy_file in npy_files:
            signal = np.load(npy_file)
            signals.append(signal)
            metadata.append({"filename": str(npy_file)})
            
        return np.array(signals), pd.DataFrame(metadata)
        
    def _load_mat_dataset(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load MATLAB format dataset."""
        import scipy.io
        
        mat_files = list(self.data_path.glob("*.mat"))
        
        if not mat_files:
            raise FileNotFoundError("No MAT files found in data path")
            
        signals = []
        metadata = []
        
        for mat_file in mat_files:
            mat_data = scipy.io.loadmat(mat_file)
            
            # Find signal data (usually largest numeric array)
            signal_key = None
            max_size = 0
            
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray):
                    if value.size > max_size:
                        max_size = value.size
                        signal_key = key
                        
            if signal_key:
                signal = mat_data[signal_key]
                signals.append(signal)
                metadata.append({"filename": str(mat_file), "signal_key": signal_key})
                
        return np.array(signals), pd.DataFrame(metadata)
        
    def _resample_signal(self, signal: np.ndarray, original_fs: int, target_fs: int) -> np.ndarray:
        """Resample signal to target frequency."""
        from scipy.signal import resample
        
        if original_fs == target_fs:
            return signal
            
        # Calculate new length
        new_length = int(len(signal) * target_fs / original_fs)
        
        # Resample each channel
        if signal.ndim == 1:
            return resample(signal, new_length)
        else:
            resampled = np.zeros((new_length, signal.shape[1]))
            for i in range(signal.shape[1]):
                resampled[:, i] = resample(signal[:, i], new_length)
            return resampled
            
    def get_dataset_info(self) -> Dict:
        """Get information about the dataset."""
        info = {
            "data_path": str(self.data_path),
            "exists": self.data_path.exists(),
            "files": []
        }
        
        if self.data_path.exists():
            # Count different file types
            file_types = {".csv": 0, ".npy": 0, ".mat": 0, ".dat": 0, ".hea": 0}
            
            for file_type in file_types.keys():
                files = list(self.data_path.glob(f"*{file_type}"))
                file_types[file_type] = len(files)
                
            info["file_types"] = file_types
            info["total_files"] = sum(file_types.values())
            
        return info