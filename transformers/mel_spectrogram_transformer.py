from sklearn.base import BaseEstimator, TransformerMixin
import librosa
import numpy as np

from config import SAMPLING_RATE, N_MELS, N_FFT, HOP_LENGTH

class MelSpectrogramTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sr=SAMPLING_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        specs = []
        for y in X:
            S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length, n_fft=self.n_fft)
            S_dB = librosa.power_to_db(S, ref=np.max)
            specs.append(S_dB)
        return specs
    