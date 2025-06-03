import cv2
import numpy as np
import os

def extract_shape_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros(7)

    c = max(contours, key=cv2.contourArea)

    moments = cv2.moments(c)
    hu_moments = cv2.HuMoments(moments)
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))

    shape_features = hu_moments.flatten()

    return shape_features

if __name__ == "__main__":
    image_dir = 'images/organik'

    if not os.path.exists(image_dir):
        print(f"Direktori citra '{image_dir}' tidak ditemukan.")
    else:
        all_shape_features = []
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print(f"Tidak ada citra yang ditemukan di direktori '{image_dir}'.")
        else:
            for img_file in image_files:
                img_path = os.path.join(image_dir, img_file)
                features = extract_shape_features(img_path)
                if features is not None:
                    all_shape_features.append(features)

            if all_shape_features:
                all_shape_features = np.array(all_shape_features)
                if all_shape_features.size > 0: 
                    np.savetxt("extracted_shape_features_organik.csv", all_shape_features, delimiter=",")
                    print("Fitur bentuk untuk organik berhasil disimpan ke 'extracted_shape_features_organik.csv'")
                print(f"Ekstraksi fitur bentuk selesai. Total fitur diekstrak: {all_shape_features.shape}")
            else:
                print("Tidak ada fitur bentuk yang berhasil diekstrak.")