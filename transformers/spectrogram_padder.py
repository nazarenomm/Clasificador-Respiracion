from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from config import MAX_LENGTH

class SpectrogramPadder(BaseEstimator, TransformerMixin):
    def __init__(self, max_length=MAX_LENGTH):
        self.max_length = max_length

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        padded = []
        for mel in X:
            pad_width = self.max_length - mel.shape[1]
            if pad_width > 0:
                mel_padded = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mel_padded = mel[:, :self.max_length]
            padded.append(mel_padded)
        return np.array(padded)
