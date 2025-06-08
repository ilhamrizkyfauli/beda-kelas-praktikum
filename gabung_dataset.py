import os
import numpy as np
import cv2

from feature_extraction_color import extract_color_features
from feature_extraction_shape import extract_shape_features
from feature_extraction_texture import extract_texture_features

# Folder gambar (pastikan struktur: images/kertas/, images/plastik/, images/organik/)
base_dir = "images"
categories = ['kertas', 'organik', 'plastik']
label_map = {'kertas': 0, 'organik': 1, 'plastik': 2}

# Untuk menyimpan data dan label
data = []
labels = []

for category in categories:
    folder = os.path.join(base_dir, category)  
    for filename in os.listdir(folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):

                filepath = os.path.join(folder, filename)
    
                # Ekstraksi fitur
                color_feat = extract_color_features(filepath)
                shape_feat = extract_shape_features(filepath)
                texture_feat = extract_texture_features(filepath)
                # Gabungkan fitur
                combined_feat = np.concatenate([color_feat, shape_feat, texture_feat])
                data.append(combined_feat)
                labels.append(label_map[category])  # ✅ label berupa angka, bukan string
         

# Simpan dataset
data = np.array(data)
labels = np.array(labels).reshape(-1, 1)

# Gunakan column_stack agar dimensi cocok
gabungan = np.column_stack((data, labels))

# Simpan ke file CSV
os.makedirs('fitur', exist_ok=True) 
np.savetxt('fitur/dataset_gabungan.csv', gabungan, delimiter=',', fmt='%f')

print("✅ Dataset gabungan berhasil dibuat")
