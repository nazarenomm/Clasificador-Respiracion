import numpy as np
import random
from sklearn.base import BaseEstimator, TransformerMixin

class NoiseAdder(BaseEstimator, TransformerMixin):
    def __init__(self, snr_min=5, snr_max=10, random_state=None):
        """
        Agrega ruido blanco a la señal manteniendo una SNR (Signal-to-Noise Ratio) aleatoria
        entre snr_min y snr_max dB.

        Parámetros:
        -----------
        snr_min : float
            SNR mínima en dB (más bajo = más ruido).
        snr_max : float
            SNR máxima en dB (más alto = menos ruido).
        random_state : int o None
            Semilla para reproducibilidad.
        """
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Aplica ruido blanco a cada señal de X.

        Parámetros:
        -----------
        X : list[np.ndarray] o np.ndarray
            Lista o array de señales (1D).

        Retorna:
        --------
        list[np.ndarray]
            Señales con ruido agregado.
        """
        noisy_signals = []
        for y in X:
            snr_db = random.uniform(self.snr_min, self.snr_max)
            rms = np.sqrt(np.mean(y**2))
            snr = 10 ** (snr_db / 20.0)
            noise_rms = rms / snr
            noise = np.random.normal(0, noise_rms, y.shape)
            noisy_signals.append(y + noise)
        return noisy_signals
