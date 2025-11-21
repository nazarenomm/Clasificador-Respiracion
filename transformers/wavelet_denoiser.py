import numpy as np
import pywt
from sklearn.base import BaseEstimator, TransformerMixin

class WaveletDenoiser(BaseEstimator, TransformerMixin):
    def __init__(self, wavelet='db4', level=4, threshold_method='soft'):
        '''
        Transformador para aplicar denoising basado en wavelets en señales de audio.
        
        Parámetros:
        - wavelet: Tipo de wavelet a utilizar.
        - level: Nivel de descomposición.
        - threshold_method: Método de umbralización ('soft' o 'hard').
        '''
        self.wavelet = wavelet
        self.level = level
        self.threshold_method = threshold_method

    def _denoise(self, signal):
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)

        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(signal)))

        coeffs_thresh = [coeffs[0]] + [
            pywt.threshold(c, value=uthresh, mode=self.threshold_method)
            for c in coeffs[1:]
        ]

        denoised = pywt.waverec(coeffs_thresh, self.wavelet)
        return denoised

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # X es una lista o array de señales 1D
        denoised_signals = []
        for x in X:
            try:
                denoised = self._denoise(x)
            except Exception:
                denoised = x  # fallback si hay error
            denoised_signals.append(denoised)
        return denoised_signals
