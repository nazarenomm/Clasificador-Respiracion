import os
import shutil

src_dir = './icbhi/ICBHI_final_database'
dst_dir = './dataset/enteros'

os.makedirs(dst_dir, exist_ok=True)

for root, _, files in os.walk(src_dir):
    for file in files:
        if file.lower().endswith('.wav'):
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_dir, file)
            shutil.copy2(src_file, dst_file)