import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os

def run_classification(features, labels):
    # Mendeteksi baris yang mengandung NaN
    nan_rows = np.isnan(features).any(axis=1)
    num_nan_samples = np.sum(nan_rows)

    if num_nan_samples > 0:
        print(f"Peringatan: Ditemukan {num_nan_samples} sampel yang mengandung NaN. Sampel ini akan dihapus.")
        features = features[~nan_rows] 
        labels = labels[~nan_rows]    
        print(f"Sisa sampel setelah penghapusan NaN: {features.shape[0]}")
    else:
        print("Tidak ada nilai NaN yang ditemukan dalam data fitur.")
    # --- Akhir bagian penanganan NaN ---

    if features.shape[0] == 0:
        print("Tidak ada sampel yang tersisa setelah menghapus NaN. Klasifikasi tidak dapat dilakukan.")
        return

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_predictions = knn_model.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    print(f"Akurasi KNN: {knn_accuracy:.2f}")

    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    print(f"Akurasi SVM: {svm_accuracy:.2f}")

if __name__ == "__main__":
    try:
        # Memuat fitur
        color_features = np.loadtxt("extracted_color_features_kertas.csv", delimiter=",")
        shape_features = np.loadtxt("extracted_shape_features_organik.csv", delimiter=",")
        texture_features = np.loadtxt("extracted_texture_features_plastik.csv", delimiter=",")

        max_dim = 768

        # Pastikan fitur adalah array 2D
        if color_features.ndim == 1:
            color_features = np.expand_dims(color_features, axis=0)
        if shape_features.ndim == 1:
            shape_features = np.expand_dims(shape_features, axis=0)
        if texture_features.ndim == 1:
            texture_features = np.expand_dims(texture_features, axis=0)

        # Padding shape_features
        padded_shape_features = np.pad(shape_features, ((0, 0), (0, max_dim - shape_features.shape[1])), 'constant', constant_values=0)

        # Padding texture_features
        padded_texture_features = np.pad(texture_features, ((0, 0), (0, max_dim - texture_features.shape[1])), 'constant', constant_values=0)

        # Membuat label
        labels_kertas = np.array([0] * len(color_features))
        labels_organik = np.array([1] * len(shape_features))
        labels_plastik = np.array([2] * len(texture_features))

        # Menggabungkan semua fitur dan label
        all_features = np.vstack((color_features, padded_shape_features, padded_texture_features))
        all_labels = np.concatenate((labels_kertas, labels_organik, labels_plastik))

        print("Memulai proses klasifikasi dengan data yang dimuat...")
        run_classification(all_features, all_labels)

    except FileNotFoundError:
        print("Pastikan semua file fitur (extracted_color_features_kertas.csv, extracted_shape_features_organik.csv, extracted_texture_features_plastik.csv) sudah dibuat dan ada di direktori yang sama.")
        print("Jalankan terlebih dahulu skrip ekstraksi fitur (color, shape, texture) untuk membuat file-file tersebut.")
    except Exception as e:
        print(f"Terjadi kesalahan saat memuat atau mengolah data: {e}")