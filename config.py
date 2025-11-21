"""
Configuraciones globales para el proyecto.
"""

# Preprocesamiento
SAMPLING_RATE = 16_000
FRAME_LENGTH = 1_024
HOP_LENGTH = 256 # puede ser menos para mayor resolucion temporal y precision en la deteccion de crepitaciones.
MAX_DURATION = 6  # segundos, duración máxima de un ciclo respiratorio (inspiración + espiración)
MIN_DURATION = 1  # segundos, duración mínima de un ciclo respiratorio (inspiración + espiración)
N_MELS = 128
N_FFT = 1_024
MAX_LENGTH = 1 + int((MAX_DURATION * SAMPLING_RATE - N_FFT) // HOP_LENGTH)
PADDING_MODE = 'center'  # 'end', 'start' o 'center'
N_MFCC = 40

# Filtros pasa banda
HIGH_CUT = 4_000
LOW_CUT = 100

# Entrenamiento
SEED = 42
TEST_SIZE = 0.2