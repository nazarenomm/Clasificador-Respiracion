from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class Normalizer(BaseEstimator, TransformerMixin):
    def __init__(self, method='rms', target_level=0.1, eps=1e-8, smooth_factor=0.01):
        self.method = method
        self.target_level = target_level
        self.eps = eps
        self.smooth_factor = smooth_factor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = []
        for x in X:
            if self.method == 'peak':
                max_amp = np.max(np.abs(x))
                x_norm = x / (max_amp + self.eps)
            elif self.method == 'rms':
                rms = np.sqrt(np.mean(x ** 2))
                x_norm = x * (self.target_level / (rms + self.eps))
            elif self.method == 'amplitude':
                env = np.convolve(np.abs(x), np.ones(int(len(x)*self.smooth_factor)), mode='same')
                env /= np.max(env) + self.eps
                x_norm = x / (env + self.eps)
            else:
                raise ValueError("Método no soportado")
            out.append(x_norm)
        return np.array(out, dtype=object)
