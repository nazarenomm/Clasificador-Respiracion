from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import librosa

from config import N_FFT, HOP_LENGTH, FRAME_LENGTH as WIN_LENGTH

class SpectralSubtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, alpha=1.5):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.alpha = alpha

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processed = []
        for y in X:
            S = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
            mag, phase = np.abs(S), np.angle(S)
            noise_mag = np.median(mag, axis=1, keepdims=True)
            mag_denoised = np.maximum(mag - self.alpha * noise_mag, 1e-10)
            S_d = mag_denoised * np.exp(1j * phase)
            y_d = librosa.istft(S_d, hop_length=self.hop_length, win_length=self.win_length)
            processed.append(y_d)
        return processed