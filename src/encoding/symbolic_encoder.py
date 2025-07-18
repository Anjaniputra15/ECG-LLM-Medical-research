import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import KMeans
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SymbolicEncoder:
    """Convert ECG signals to symbolic token representations."""
    
    def __init__(self, vocab_size: int = 1000, segment_length: int = 128):
        self.vocab_size = vocab_size
        self.segment_length = segment_length
        self.logger = logger
        self.symbol_map = {}
        self.is_fitted = False
        
    def fit(self, ecg_signals: np.ndarray) -> 'SymbolicEncoder':
        """Fit the symbolic encoder to ECG signals.
        
        Args:
            ecg_signals: Array of ECG signals for training
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Fitting symbolic encoder with {len(ecg_signals)} signals...")
        
        # Extract features from all signals
        features = self._extract_features(ecg_signals)
        
        # Cluster features to create vocabulary
        self.logger.info(f"Clustering features into {self.vocab_size} symbols...")
        self.kmeans = KMeans(n_clusters=self.vocab_size, random_state=42)
        self.kmeans.fit(features)
        
        # Create symbol mapping
        self.symbol_map = {i: f"SYM_{i:04d}" for i in range(self.vocab_size)}
        
        self.is_fitted = True
        self.logger.info("Symbolic encoder fitted successfully")
        
        return self
        
    def encode(self, ecg_signal: np.ndarray) -> List[str]:
        """Encode ECG signal to symbolic tokens.
        
        Args:
            ecg_signal: ECG signal to encode
            
        Returns:
            List of symbolic tokens
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before encoding")
            
        # Segment the signal
        segments = self._segment_signal(ecg_signal)
        
        # Extract features from segments
        features = self._extract_segment_features(segments)
        
        # Assign symbols based on clustering
        symbols = self.kmeans.predict(features)
        
        # Convert to symbolic tokens
        tokens = [self.symbol_map[symbol] for symbol in symbols]
        
        self.logger.debug(f"Encoded signal into {len(tokens)} tokens")
        return tokens
        
    def encode_batch(self, ecg_signals: List[np.ndarray]) -> List[List[str]]:
        """Encode multiple ECG signals.
        
        Args:
            ecg_signals: List of ECG signals
            
        Returns:
            List of token sequences
        """
        return [self.encode(signal) for signal in ecg_signals]
        
    def decode(self, tokens: List[str]) -> np.ndarray:
        """Decode symbolic tokens back to approximate ECG signal.
        
        Args:
            tokens: List of symbolic tokens
            
        Returns:
            Reconstructed ECG signal
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before decoding")
            
        # Get cluster centers for each token
        reconstructed_segments = []
        
        for token in tokens:
            # Find symbol ID
            symbol_id = None
            for sid, stoken in self.symbol_map.items():
                if stoken == token:
                    symbol_id = sid
                    break
                    
            if symbol_id is not None:
                # Get cluster center
                cluster_center = self.kmeans.cluster_centers_[symbol_id]
                
                # Reconstruct segment from features (simplified)
                segment = self._features_to_segment(cluster_center)
                reconstructed_segments.append(segment)
                
        # Concatenate segments
        if reconstructed_segments:
            reconstructed_signal = np.concatenate(reconstructed_segments)
        else:
            reconstructed_signal = np.zeros(self.segment_length)
            
        return reconstructed_signal
        
    def _extract_features(self, ecg_signals: np.ndarray) -> np.ndarray:
        """Extract features from ECG signals for clustering.
        
        Args:
            ecg_signals: Array of ECG signals
            
        Returns:
            Feature matrix
        """
        all_features = []
        
        for signal in ecg_signals:
            segments = self._segment_signal(signal)
            features = self._extract_segment_features(segments)
            all_features.extend(features)
            
        return np.array(all_features)
        
    def _segment_signal(self, ecg_signal: np.ndarray) -> List[np.ndarray]:
        """Segment ECG signal into fixed-length segments.
        
        Args:
            ecg_signal: ECG signal
            
        Returns:
            List of signal segments
        """
        if ecg_signal.ndim == 1:
            signal_length = len(ecg_signal)
        else:
            signal_length = ecg_signal.shape[0]
            
        segments = []
        
        for i in range(0, signal_length, self.segment_length):
            end_idx = min(i + self.segment_length, signal_length)
            segment = ecg_signal[i:end_idx]
            
            # Pad if necessary
            if len(segment) < self.segment_length:
                if ecg_signal.ndim == 1:
                    segment = np.pad(segment, (0, self.segment_length - len(segment)), 'constant')
                else:
                    pad_length = self.segment_length - segment.shape[0]
                    segment = np.pad(segment, ((0, pad_length), (0, 0)), 'constant')
                    
            segments.append(segment)
            
        return segments
        
    def _extract_segment_features(self, segments: List[np.ndarray]) -> np.ndarray:
        """Extract features from signal segments.
        
        Args:
            segments: List of signal segments
            
        Returns:
            Feature matrix
        """
        features = []
        
        for segment in segments:
            # Statistical features
            if segment.ndim == 1:
                feature_vector = self._extract_single_lead_features(segment)
            else:
                # Multi-lead features
                lead_features = []
                for lead_idx in range(segment.shape[1]):
                    lead_feature = self._extract_single_lead_features(segment[:, lead_idx])
                    lead_features.extend(lead_feature)
                feature_vector = np.array(lead_features)
                
            features.append(feature_vector)
            
        return np.array(features)
        
    def _extract_single_lead_features(self, segment: np.ndarray) -> np.ndarray:
        """Extract features from a single-lead segment.
        
        Args:
            segment: Single-lead ECG segment
            
        Returns:
            Feature vector
        """
        features = []
        
        # Statistical features
        features.extend([
            np.mean(segment),
            np.std(segment),
            np.min(segment),
            np.max(segment),
            np.median(segment),
            np.percentile(segment, 25),
            np.percentile(segment, 75),
            np.var(segment)
        ])
        
        # Morphological features
        features.extend([
            np.sum(np.abs(np.diff(segment))),  # Total variation
            np.sum(segment > 0),  # Positive samples
            np.sum(segment < 0),  # Negative samples
            np.argmax(segment),  # Peak location
            np.argmin(segment),  # Trough location
        ])
        
        # Frequency domain features (simplified)
        fft_vals = np.abs(np.fft.fft(segment))
        features.extend([
            np.sum(fft_vals[:len(fft_vals)//4]),  # Low frequency power
            np.sum(fft_vals[len(fft_vals)//4:len(fft_vals)//2]),  # High frequency power
            np.argmax(fft_vals[:len(fft_vals)//2]),  # Dominant frequency
        ])
        
        return np.array(features)
        
    def _features_to_segment(self, features: np.ndarray) -> np.ndarray:
        """Convert features back to signal segment (simplified reconstruction).
        
        Args:
            features: Feature vector
            
        Returns:
            Reconstructed segment
        """
        # This is a simplified reconstruction
        # In practice, you'd want a more sophisticated method
        mean_val = features[0]
        std_val = features[1]
        
        # Generate segment with similar statistics
        segment = np.random.normal(mean_val, std_val, self.segment_length)
        
        return segment
        
    def get_vocabulary(self) -> Dict[int, str]:
        """Get the symbol vocabulary.
        
        Returns:
            Dictionary mapping symbol IDs to tokens
        """
        return self.symbol_map
        
    def get_vocab_size(self) -> int:
        """Get vocabulary size.
        
        Returns:
            Vocabulary size
        """
        return self.vocab_size
        
    def save_encoder(self, filepath: str):
        """Save the encoder to file.
        
        Args:
            filepath: Path to save encoder
        """
        import pickle
        
        encoder_data = {
            'vocab_size': self.vocab_size,
            'segment_length': self.segment_length,
            'symbol_map': self.symbol_map,
            'kmeans': self.kmeans,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(encoder_data, f)
            
        self.logger.info(f"Encoder saved to {filepath}")
        
    def load_encoder(self, filepath: str):
        """Load encoder from file.
        
        Args:
            filepath: Path to load encoder from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            encoder_data = pickle.load(f)
            
        self.vocab_size = encoder_data['vocab_size']
        self.segment_length = encoder_data['segment_length']
        self.symbol_map = encoder_data['symbol_map']
        self.kmeans = encoder_data['kmeans']
        self.is_fitted = encoder_data['is_fitted']
        
        self.logger.info(f"Encoder loaded from {filepath}")
        
    def get_encoding_stats(self, tokens: List[str]) -> Dict:
        """Get statistics about the encoded tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Dictionary with encoding statistics
        """
        from collections import Counter
        
        token_counts = Counter(tokens)
        
        stats = {
            'total_tokens': len(tokens),
            'unique_tokens': len(token_counts),
            'vocabulary_usage': len(token_counts) / self.vocab_size,
            'most_common_tokens': token_counts.most_common(10),
            'average_token_frequency': np.mean(list(token_counts.values())),
            'token_entropy': self._calculate_entropy(token_counts)
        }
        
        return stats
        
    def _calculate_entropy(self, token_counts: Dict) -> float:
        """Calculate entropy of token distribution.
        
        Args:
            token_counts: Token frequency counter
            
        Returns:
            Entropy value
        """
        total_tokens = sum(token_counts.values())
        probabilities = [count / total_tokens for count in token_counts.values()]
        
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        return entropy