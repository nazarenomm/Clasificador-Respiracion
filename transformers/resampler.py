from sklearn.base import BaseEstimator, TransformerMixin
import librosa

from config import SAMPLING_RATE

class Resampler(BaseEstimator, TransformerMixin):
    def __init__(self, sr_target=SAMPLING_RATE):
        self.sr_target = sr_target

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        resampled = []
        for path in X:
            y, sr = librosa.load(path, sr=None)
            if sr != self.sr_target:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sr_target)
            resampled.append(y)
        return resampled
