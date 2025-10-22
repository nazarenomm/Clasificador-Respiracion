from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import librosa

from config import SAMPLING_RATE, FRAME_LENGTH, HOP_LENGTH, N_MFCC, N_FFT

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, sr=SAMPLING_RATE, n_mfcc=N_MFCC, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, n_fft=N_FFT):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.frame_length = frame_length

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
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.frame_length, hop_length=self.hop_length).mean()

        # --- Short-Time Energy ---
        energy = np.array([
            np.sum(np.abs(y[i:i+self.frame_length]**2))
            for i in range(0, len(y), self.hop_length)
        ])
        energy_mean = energy.mean()

        # --- Spectral Centroid ---
        sc = librosa.feature.spectral_centroid(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length).mean()

        # --- Spectral Roll-off ---
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length).mean()

        # --- Band Energy Ratio (BER): energía en banda baja vs total ---
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        low_band = freqs < 1000
        ber = S[low_band, :].sum() / S.sum()

        # --- Spectral Flatness ---
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=self.n_fft, hop_length=self.hop_length).mean()

        return np.hstack([
            mfccs_mean,
            zcr,
            energy_mean,
            sc,
            rolloff,
            ber,
            flatness
        ])