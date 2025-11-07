import librosa
import soundfile as sf
import os

input_folder = "./dataset/ruido_largo"
output_folder = "./dataset/ruido"
SAMPLING_RATE = 16_000
segment_duration = 30  # segundos

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.endswith(".wav"):
        continue

    filepath = os.path.join(input_folder, filename)
    audio, sr = librosa.load(filepath, sr=SAMPLING_RATE)

    samples_per_segment = segment_duration * SAMPLING_RATE
    num_segments = len(audio) // samples_per_segment

    for i in range(num_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment
        segment = audio[start:end]

        segment_filename = f"{os.path.splitext(filename)[0]}_part{i+1}.wav"
        sf.write(os.path.join(output_folder, segment_filename), segment, SAMPLING_RATE)
