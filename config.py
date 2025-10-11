"""
Configuraciones globales para el proyecto.
"""

# Audio
SAMPLING_RATE = 16_000
FRAME_LENGTH = 1_024
HOP_LENGTH = 512
HIGH_CUT = 4_000
LOW_CUT = 0

# Espectrograma
N_MELS = 64
N_FFT = 1_024
MAX_LENGTH = 245  # TODO: Ajustar

# Entrenamiento
SEED = 42
TEST_SIZE = 0.2