import cv2
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops

def extract_texture_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]

    texture_features = np.array([contrast, dissimilarity, homogeneity, energy, correlation, asm])

    return texture_features

if __name__ == "__main__":
    image_dir = 'images/plastik' 

    if not os.path.exists(image_dir):
        print(f"Direktori citra '{image_dir}' tidak ditemukan.")
    else:
        all_texture_features = []
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

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
                if all_texture_features.size > 0:
                    np.savetxt("extracted_texture_features_plastik.csv", all_texture_features, delimiter=",")
                    print("Fitur tekstur untuk plastik berhasil disimpan ke 'extracted_texture_features_plastik.csv'")
                print(f"Ekstraksi fitur tekstur selesai. Total fitur diekstrak: {all_texture_features.shape}")
            else:
                print("Tidak ada fitur tekstur yang berhasil diekstrak.")