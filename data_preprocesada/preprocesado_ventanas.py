import os
import pandas as pd
import numpy as np
import soundfile as sf

from config import WINDOW_SIZE

input_audio_dir = "./dataset/icbhi/ICBHI_final_database"
input_events_dir = "./dataset/icbhi/events"
output_dir = "./data_preprocesada/ventanas"
os.makedirs(output_dir, exist_ok=True)

window_size = WINDOW_SIZE

data_rows = []

# Función para leer eventos
def load_events(txt_path):
    events = []
    if not os.path.exists(txt_path):
        return events
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, label = float(parts[0]), float(parts[1]), parts[2].lower()
                events.append((start, end, label))
    return events

# Recorremos los .wav
for root, _, files in os.walk(input_audio_dir):
    for file in files:
        if not file.lower().endswith(".wav"):
            continue

        wav_path = os.path.join(root, file)
        events_path = os.path.join(input_events_dir, file.replace(".wav", "_events.txt"))

        try:
            audio, sr = sf.read(wav_path)
        except Exception as e:
            print(f"Error al leer {wav_path}: {e}")
            continue

        duration = len(audio) / sr
        events = load_events(events_path)

        # Creamos las ventanas
        n_windows = int(np.ceil(duration / window_size))

        for i in range(n_windows):
            start_time = i * window_size
            end_time = min((i + 1) * window_size, duration)

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment = audio[start_sample:end_sample]

            # Determinamos la etiqueta (normal, crackles, wheezes, both. Para posible clasificacion multiclase)
            window_labels = set()
            for ev_start, ev_end, ev_label in events:
                overlap = not (ev_end <= start_time or ev_start >= end_time)
                if overlap:
                    window_labels.add(ev_label)

            if not window_labels:
                label = "normal"
            elif len(window_labels) == 1:
                label = list(window_labels)[0]
            else:
                label = "both"

            # Guardamos la ventana
            new_name = f"{os.path.splitext(file)[0]}_win{i:03d}.wav"
            new_path = os.path.join(output_dir, new_name)
            sf.write(new_path, segment, sr)

            data_rows.append({"filename": new_name, "label": label})

# Guardamos el CSV con las eqtiquetas
df = pd.DataFrame(data_rows)
df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)