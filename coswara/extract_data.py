import os
import glob
import tarfile

'''
This script creates a folder "Extracted_data" inside which it extracts all the wav files in the directories date-wise
'''

coswara_data_dir = os.path.abspath('.')  # Local Path of iiscleap/Coswara-Data Repo
extracted_data_dir = os.path.join(coswara_data_dir, 'Extracted_data')

if not os.path.exists(coswara_data_dir):
    raise Exception("Check the Coswara dataset directory!")

if not os.path.exists(extracted_data_dir):
    os.makedirs(extracted_data_dir)  # Creates the Extracted_data folder if it doesn't exist

# Directorios que ya fueron extraídos
dirs_extracted = set(map(os.path.basename, glob.glob(f'{extracted_data_dir}/202*')))
# Directorios que contienen los .tar.gz.*
dirs_all = set(map(os.path.basename, glob.glob(f'{coswara_data_dir}/202*')))

# Falta por extraer
dirs_to_extract = list(set(dirs_all) - dirs_extracted)

for d in dirs_to_extract:
    part_files = sorted(glob.glob(f"{coswara_data_dir}/{d}/*.tar.gz.*"))
    if not part_files:
        print(f"No se encontraron fragmentos en {d}, lo salto.")
        continue

    combined_path = os.path.join(extracted_data_dir, f"{d}.tar.gz")

    # Concatenar los fragmentos en un solo archivo
    print(f"Unificando fragmentos en {combined_path}...")
    with open(combined_path, "wb") as f_out:
        for pf in part_files:
            with open(pf, "rb") as f_in:
                f_out.write(f_in.read())

    # Extraer el tar.gz
    print(f"Extrayendo {combined_path} en {extracted_data_dir}...")
    with tarfile.open(combined_path, "r:gz") as tar:
        tar.extractall(path=extracted_data_dir)

    print(f"Directorio {d} extraído en {extracted_data_dir}")

print("✅ Extraction process complete!")