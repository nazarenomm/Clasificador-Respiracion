from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import librosa

from config import SAMPLING_RATE, FRAME_LENGTH, HOP_LENGTH

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, sr=SAMPLING_RATE, n_mfcc=13):
        self.sr = sr
        self.n_mfcc = n_mfcc

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for y in X:
            feats = self._extract_features(y)
            features.append(feats)
        return np.array(features)

    def _extract_features(self, y):
        # --- MFCCs ---
        mfccs = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
        mfccs_mean = mfccs.mean(axis=1)

        # --- ZCR ---
        zcr = librosa.feature.zero_crossing_rate(y).mean()

        # --- Short-Time Energy ---
        energy = np.array([
            np.sum(np.abs(y[i:i+FRAME_LENGTH]**2))
            for i in range(0, len(y), HOP_LENGTH)
        ])
        energy_mean = energy.mean()

        # --- Spectral Centroid ---
        sc = librosa.feature.spectral_centroid(y=y, sr=self.sr).mean()

        # --- Spectral Roll-off ---
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr).mean()

        # --- Band Energy Ratio (BER): energía en banda baja vs total ---
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=self.sr)
        low_band = freqs < 1000
        ber = S[low_band, :].sum() / S.sum()

        # --- Spectral Flatness ---
        flatness = librosa.feature.spectral_flatness(y=y).mean()

        return np.hstack([
            mfccs_mean,
            zcr,
            energy_mean,
            sc,
            rolloff,
            ber,
            flatness
        ])