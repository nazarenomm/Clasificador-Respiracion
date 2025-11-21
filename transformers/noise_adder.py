import numpy as np
import random
import soundfile as sf
from sklearn.base import BaseEstimator, TransformerMixin

from config import SAMPLING_RATE, SEED

class NoiseAdder(BaseEstimator, TransformerMixin):
    def __init__(self, noise_files, snr_min=5, snr_max=10, mode='pool', random_state=SEED):
        """
        Agrega ruido real o blanco a las señales según el modo seleccionado.

        Parámetros:
        noise_files : Lista de rutas a archivos .wav con grabaciones de ruido (mono, ya resampleados).
        snr_min, snr_max : Rango de SNR (en dB) para el escalado del ruido.
        mode : {'pool', 'white'}
            'pool' usa segmentos aleatorios de los audios de ruido.
            'white' genera ruido blanco.
        random_state : Semilla para reproducibilidad.
        """
        self.noise_files = noise_files
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.mode = mode
        self.random_state = random_state
        
        random.seed(random_state)

        # Pre-carga los ruidos
        self.noises = []
        if self.mode == 'pool':
            for path in noise_files:
                noise, sr = sf.read(path)
                if sr != SAMPLING_RATE:
                    raise ValueError(f"Sample rate inesperado en {path}: {sr} (se esperaba {SAMPLING_RATE})")
                if noise.ndim > 1:
                    noise = np.mean(noise, axis=1)
                self.noises.append(noise.astype(np.float32))

    def fit(self, X, y=None):
        return self

    def _get_noise_segment(self, length):
        """Obtiene un segmento de ruido del largo deseado."""
        if self.mode == 'white' or not self.noises:
            return np.random.normal(0, 1, length).astype(np.float32)

        # Elige un ruido al azar del pool
        noise = random.choice(self.noises)
        if len(noise) < length:
            # si es más corto, lo repite
            reps = int(np.ceil(length / len(noise)))
            noise = np.tile(noise, reps)

        start = random.randint(0, len(noise) - length)
        seg = noise[start:start + length]
        return seg.astype(np.float32)

    def transform(self, X, y=None):
        """Aplica el ruido a cada señal en X."""
        noisy_signals = []
        for signal in X:
            length = len(signal)
            snr_db = random.uniform(self.snr_min, self.snr_max)

            # RMS de la señal y ruido target según SNR
            rms_signal = np.sqrt(np.mean(signal ** 2)) + 1e-12
            snr = 10 ** (snr_db / 20.0)
            noise_rms_target = rms_signal / snr

            # Obtiene el ruido (segmento o blanco)
            noise = self._get_noise_segment(length)
            # Normaliza ruido y escala a RMS deseado
            noise = noise / (np.sqrt(np.mean(noise ** 2)) + 1e-12) * noise_rms_target

            # Pequeño desplazamiento aleatorio del ruido
            shift = random.randint(0, int(0.1 * length))
            noise = np.roll(noise, shift)

            noisy_signals.append(signal + noise)
        return noisy_signals
