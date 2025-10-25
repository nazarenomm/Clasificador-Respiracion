from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from config import MAX_LENGTH, PADDING_MODE

class SpectrogramPadder(BaseEstimator, TransformerMixin):
    def __init__(self, max_length=MAX_LENGTH, mode=PADDING_MODE, pad_value=None):
        """
        Parameters
        ----------
        max_length : int
            Longitud máxima del eje temporal del espectrograma.
        mode : str
            Estrategia de padding: 'end', 'start' o 'center'.
        """
        self.max_length = max_length
        self.pad_value = pad_value
        self.mode = mode.lower()
        if self.mode not in ['end', 'start', 'center']:
            raise ValueError("mode must be 'end', 'start', or 'center'")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        padded = []
        for mel in X:
            current_len = mel.shape[1]
            pad_width = self.max_length - current_len

            if pad_width > 0:
                if self.mode == 'end':
                    # Padding al final
                    pad = ((0, 0), (0, pad_width))
                elif self.mode == 'start':
                    # Padding al inicio
                    pad = ((0, 0), (pad_width, 0))
                elif self.mode == 'center':
                    # Padding repartido
                    pad_left = pad_width // 2
                    pad_right = pad_width - pad_left
                    pad = ((0, 0), (pad_left, pad_right))
                    
                const_val = self.pad_value if self.pad_value is not None else mel.min()
                mel_padded = np.pad(mel, pad, mode='constant', constant_values=const_val)
            else:
                # Recortar si excede max_length
                if self.mode == 'end':
                    mel_padded = mel[:, :self.max_length]
                elif self.mode == 'start':
                    mel_padded = mel[:, -self.max_length:]
                elif self.mode == 'center':
                    start = (current_len - self.max_length) // 2
                    mel_padded = mel[:, start:start + self.max_length]

            padded.append(mel_padded)

        return np.array(padded)