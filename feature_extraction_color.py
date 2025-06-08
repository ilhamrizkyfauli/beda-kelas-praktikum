import cv2
import numpy as np
import os

def extract_color_features(image_path):
    img = cv2.imread(image_path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (200, 200))  # Optional: resize agar konsisten
    
    # --- FUNGSI EXTRACT LAB FEATURES DIINTEGRASIKAN DI SINI ---
    # Convert RGB ke Lab
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    # Statistik mean dan std (L, a, b)
    mean_lab = cv2.mean(img_lab)[:3]
    std_lab = np.std(img_lab.reshape(-1, 3), axis=0)
    lab_statistik = list(mean_lab) + list(std_lab)

    # Histogram Lab 8 bin per channel
    hist_l = cv2.calcHist([img_lab], [0], None, [8], [0, 256]).flatten()
    hist_a = cv2.calcHist([img_lab], [1], None, [8], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_lab], [2], None, [8], [0, 256]).flatten()
    hist_l = cv2.normalize(hist_l, hist_l).flatten()
    hist_a = cv2.normalize(hist_a, hist_a).flatten()
    hist_b = cv2.normalize(hist_b, hist_b).flatten()
    lab_histogram = list(hist_l) + list(hist_a) + list(hist_b)

    # --- RGB FEATURES ---
    # Statistik mean dan std (R,G,B)
    mean_rgb = cv2.mean(img_rgb)[:3]
    std_rgb = np.std(img_rgb.reshape(-1, 3), axis=0)
    fitur_rgb_statistik = list(mean_rgb) + list(std_rgb)  # 6 fitur

    # Histogram warna RGB (8-bin per channel)
    hist_r = cv2.calcHist([img_rgb], [0], None, [8], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_rgb], [1], None, [8], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_rgb], [2], None, [8], [0, 256]).flatten()
    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_b = cv2.normalize(hist_b, hist_b).flatten()
    fitur_rgb_histogram = list(hist_r) + list(hist_g) + list(hist_b)  # 24 fitur

    # Rasio warna putih pada RGB
    mask_white = cv2.inRange(img_rgb, (200, 200, 200), (255, 255, 255))
    white_ratio = np.sum(mask_white > 0) / mask_white.size  # 1 fitur

    # --- HSV FEATURES ---
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Statistik mean dan std (H,S,V)
    mean_hsv = cv2.mean(img_hsv)[:3]
    std_hsv = np.std(img_hsv.reshape(-1, 3), axis=0)
    fitur_hsv_statistik = list(mean_hsv) + list(std_hsv)  # 6 fitur

    # Histogram HSV: H=8 bins (0-180), S=4 bins, V=4 bins
    hist_h = cv2.calcHist([img_hsv], [0], None, [8], [0, 180]).flatten()
    hist_s = cv2.calcHist([img_hsv], [1], None, [4], [0, 256]).flatten()
    hist_v = cv2.calcHist([img_hsv], [2], None, [4], [0, 256]).flatten()
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    fitur_hsv_histogram = list(hist_h) + list(hist_s) + list(hist_v)  # 16 fitur

    # Gabungkan semua fitur
    color_features = (
        fitur_rgb_statistik + 
        fitur_rgb_histogram + 
        [white_ratio] + 
        fitur_hsv_statistik + 
        fitur_hsv_histogram +
        lab_statistik +
        lab_histogram
    )  # Total fitur: 6 + 24 + 1 + 6 + 16 + 6 + 24 = 83 fitur

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

        all_color_features = []
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            features = extract_color_features(img_path)
            if features is not None:
                all_color_features.append(features)

        if all_color_features:
            all_color_features = np.array(all_color_features)
            print("--------------------------------------------")
            print(f"Ekstraksi fitur warna selesai. Total fitur diekstrak: {all_color_features.shape}")
            np.savetxt("fitur\extracted_color_features_kertas.csv", all_color_features, delimiter=",")
            print("Fitur warna untuk kertas berhasil disimpan ke 'extracted_color_features_kertas.csv'")
        else:
            print("Tidak ada fitur warna yang berhasil diekstrak.")
