from sklearn.base import BaseEstimator, TransformerMixin
import librosa
import numpy as np
from scipy.signal import butter, filtfilt

from config import SAMPLING_RATE, HIGH_CUT, LOW_CUT

class BandPassFilter(BaseEstimator, TransformerMixin):
    def __init__(self, lowcut=LOW_CUT, highcut=HIGH_CUT, order=6, sr=SAMPLING_RATE):
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.sr = sr

    def _filter(self, y, sr):
        nyq = 0.5 * sr
        if self.lowcut == 0:
            normal_cut = self.highcut / nyq
            b, a = butter(self.order, normal_cut, btype='low')
        else:
            normal_low = self.lowcut / nyq
            normal_high = self.highcut / nyq
            b, a = butter(self.order, [normal_low, normal_high], btype='band')
        return filtfilt(b, a, y)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        filtered = [self._filter(y, self.sr) for y in X]
        return filtered