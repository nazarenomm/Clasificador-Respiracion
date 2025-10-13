"""
Configuraciones globales para el proyecto.
"""

# Audio
SAMPLING_RATE = 16_000
FRAME_LENGTH = 1_024
HOP_LENGTH = 512 # puede ser 256 o 128 para mayor resolucion temporal y precision en la deteccion de crepitaciones. Aumenta el tamaño del espectrograma y el tiempo de computo.
MAX_DURATION = 6  # segundos

# Filtros pasa banda
HIGH_CUT = 4_000
LOW_CUT = 100 # supuestamente las frecuencias de interes estan entre 250-350 y 2000-4000 Hz

# Mel Espectrograma
N_MELS = 128
N_FFT = 1_024
MAX_LENGTH = 1 + int((MAX_DURATION * SAMPLING_RATE - N_FFT) // HOP_LENGTH)

# Entrenamiento
SEED = 42
TEST_SIZE = 0.2