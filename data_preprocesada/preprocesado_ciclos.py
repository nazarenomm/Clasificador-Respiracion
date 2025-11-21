import os
import soundfile as sf
import pandas as pd

# Paths
DATASET_DIR = "./dataset/icbhi/ICBHI_final_database"
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "notaciones")
OUTPUT_DIR = "./data_preprocesada/ciclos"
CSV_PATH = "./data_preprocesada/ciclos/metadata.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Cargar datos demográficos
demographic_path = os.path.join(ANNOTATIONS_DIR, "demographic.txt")
demographic_df = pd.read_csv(
    demographic_path,
    sep="\t",
    header=None,
    names=["patient", "age", "sex", "adult_bmi", "child_weight", "child_height"]
)

# Cargar diagnosis
diagnosis_path = os.path.join(ANNOTATIONS_DIR, "diagnosis.txt")
diagnosis_df = pd.read_csv(
    diagnosis_path,
    sep="\t",
    header=None,
    names=["patient", "diagnosis"]
)

# Merge
demographics_dict = demographic_df.set_index("patient").to_dict(orient="index")
diagnosis_dict = diagnosis_df.set_index("patient")["diagnosis"].to_dict()

# audios
rows = []

for file in os.listdir(DATASET_DIR):
    if file.endswith(".wav"):
        wav_path = os.path.join(DATASET_DIR, file)
        txt_path = wav_path.replace(".wav", ".txt")

        if not os.path.exists(txt_path):
            continue

        # Cargamos el audio
        audio, sr = sf.read(wav_path)

        # Parseamos el nombre del archivo
        base_name = os.path.splitext(file)[0]
        patient, rec_idx, location, mode, instrument = base_name.split("_")
        patient_id = int(patient)

        # Datos demográficos
        demo = demographics_dict.get(patient_id, {})
        diag = diagnosis_dict.get(patient_id, "NA")

        # Leemos las anotaciones
        with open(txt_path, "r") as f:
            lines = f.readlines()

        # Procesamos cada ciclo
        for i, line in enumerate(lines):
            start, end, crackles, wheezes = line.strip().split("\t")
            start, end = float(start), float(end)
            crackles, wheezes = int(crackles), int(wheezes)

            # Convertimos a muestras
            start_sample = int(start * sr)
            end_sample = int(end * sr)

            # Extraemos el ciclo
            cycle_audio = audio[start_sample:end_sample]

            # Nombre archivo ciclo
            cycle_id = f"{base_name}_cycle{i+1}"
            out_wav = os.path.join(OUTPUT_DIR, f"{cycle_id}.wav")

            # Guardamos el ciclo
            sf.write(out_wav, cycle_audio, sr)

            # Agregamos metadata
            rows.append({
                "patient": patient_id,
                "recording_index": rec_idx,
                "location": location,
                "mode": mode,
                "instrument": instrument,
                "cycle_number": i+1,
                "start_sec": start,
                "end_sec": end,
                "duration_sec": end - start,
                "crackles": crackles,
                "wheezes": wheezes,
                "cycle_wav_file": f"{cycle_id}.wav",
                "wav_base_file": file,
                # Demográficos y diagnóstico
                "age": demo.get("age", "NA"),
                "sex": demo.get("sex", "NA"),
                "adult_bmi": demo.get("adult_bmi", "NA"),
                "child_weight": demo.get("child_weight", "NA"),
                "child_height": demo.get("child_height", "NA"),
                "diagnosis": diag
            })

# Guardamos el CSV con la metadata
df = pd.DataFrame(rows)
df.to_csv(CSV_PATH, index=False)
