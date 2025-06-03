import cv2
import numpy as np
import os

def extract_color_features(image_path):
    print(f"Mencoba membaca citra: {image_path}") 
    img = cv2.imread(image_path)
    if img is None:
        print(f"Gagal membaca citra: {image_path}") 
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256])

    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_b = cv2.normalize(hist_b, hist_b).flatten()

    color_features = np.concatenate((hist_r, hist_g, hist_b))

    return color_features

if __name__ == "__main__":
    image_dir = 'images/kertas'

    if not os.path.exists(image_dir):
        print(f"Direktori citra '{image_dir}' tidak ditemukan.")
    else:
        all_files_in_dir = os.listdir(image_dir) 
        print(f"Jumlah total file di direktori '{image_dir}': {len(all_files_in_dir)}")

        image_files = [f for f in all_files_in_dir if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Jumlah file citra yang terdeteksi setelah filter: {len(image_files)}")

        if not image_files:
            print(f"Tidak ada citra yang ditemukan di direktori '{image_dir}'.")
        else:
            all_color_features = [] 
            for img_file in image_files:
                img_path = os.path.join(image_dir, img_file)
                features = extract_color_features(img_path)
                if features is not None:
                    all_color_features.append(features)
                else:
                    print(f"SKIP: Citra {img_file} tidak diproses karena gagal dibaca.")

            if all_color_features:
                all_color_features = np.array(all_color_features)
                print(f"Ekstraksi fitur warna selesai. Total fitur diekstrak: {all_color_features.shape}")
                np.savetxt("extracted_color_features_kertas.csv", all_color_features, delimiter=",")
                print("Fitur warna untuk kertas berhasil disimpan ke 'extracted_color_features_kertas.csv'")
            else:
                print("Tidak ada fitur warna yang berhasil diekstrak.")