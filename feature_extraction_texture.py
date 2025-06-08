import cv2
import numpy as np
import os
import pywt
import mahotas
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.ndimage import uniform_filter, sobel

def extract_texture_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Gagal membaca citra: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- GLCM ---
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]

    # --- LBP ---
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    n_bins = int(n_points + 2)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    # --- Gabor Features ---
    gabor_features = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        kernel = cv2.getGaborKernel((21, 21), 8.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        fimg = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        gabor_features.append(fimg.mean())
        gabor_features.append(fimg.var())

    # --- Wavelet Features ---
    coeffs = pywt.wavedec2(gray, 'db1', level=2)
    wavelet_features = []
    for level in coeffs[1:]:
        for arr in level:
            arr = np.array(arr)
            energy = np.sum(arr**2)
            entropy = -np.sum(arr * np.log2(np.abs(arr) + 1e-6))
            wavelet_features.append(energy)
            wavelet_features.append(entropy)

    # --- Haralick Features ---
    haralick_feats = mahotas.features.haralick(gray, return_mean=True).tolist()  # 13 fitur

    # --- Tamura Features ---
    gray_f = gray.astype(np.float32)
    local_mean = uniform_filter(gray_f, size=16)
    diff = np.abs(gray_f - local_mean)
    coarseness = np.mean(diff)
    contrast = np.std(gray_f)
    dx = sobel(gray_f, axis=0)
    dy = sobel(gray_f, axis=1)
    gradient_direction = np.arctan2(dy, dx)
    directionality = np.var(gradient_direction)
    tamura_features = [coarseness, contrast, directionality]

    # Gabungkan semua fitur
    texture_features = np.hstack([
        [contrast, dissimilarity, homogeneity, energy, correlation, asm],  # GLCM
        lbp_hist,                     # LBP
        gabor_features,              # Gabor
        wavelet_features,            # Wavelet
        haralick_feats,              # Haralick
        tamura_features              # Tamura
    ])

    return texture_features

# --- Eksekusi jika dijalankan langsung ---
if __name__ == "__main__":
    image_dir = 'images/plastik'

    if not os.path.exists(image_dir):
        print(f"Direktori citra '{image_dir}' tidak ditemukan.")
    else:
        all_texture_features = []
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print(f"Tidak ada citra yang ditemukan di direktori '{image_dir}'.")
        else:
            for img_file in image_files:
                img_path = os.path.join(image_dir, img_file)
                features = extract_texture_features(img_path)
                if features is not None:
                    all_texture_features.append(features)

            if all_texture_features:
                all_texture_features = np.array(all_texture_features)
                np.savetxt("fitur/extracted_texture_features_plastik.csv", all_texture_features, delimiter=",")
                print("Fitur tekstur untuk plastik berhasil disimpan ke 'extracted_texture_features_plastik.csv'")
                print(f"Total fitur: {all_texture_features.shape[1]} | Total citra: {all_texture_features.shape[0]}")
