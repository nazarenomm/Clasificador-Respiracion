from sklearn.base import BaseEstimator, TransformerMixin
import librosa
import numpy as np
from scipy.signal import butter, filtfilt

from config import SAMPLING_RATE, HIGH_CUT, LOW_CUT

class AudioFilterResampler(BaseEstimator, TransformerMixin):
    def __init__(self, sr_target=SAMPLING_RATE, highcut=HIGH_CUT, lowcut=LOW_CUT, order=6):
        self.sr_target = sr_target
        self.highcut = highcut
        self.lowcut = lowcut
        self.order = order

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

    def _filter_and_resample(self, path):
        y, sr = librosa.load(path, sr=None)
        y = self._filter(y, sr)
        if sr != self.sr_target:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr_target)
        return y, self.sr_target

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processed = []
        for path in X:
            y, sr = self._filter_and_resample(path)
            processed.append(y)
        return processed