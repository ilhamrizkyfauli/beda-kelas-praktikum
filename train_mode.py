import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load dataset gabungan (fitur + label)
data = np.loadtxt('fitur\dataset_gabungan.csv', delimiter=',')

X = data[:, :-1]  # fitur (semua kolom kecuali kolom terakhir)
y = data[:, -1].astype(int)  # label

# Bagi data latih dan uji (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Buat model SVM dan KNN
svm_model = SVC(probability=True, kernel='rbf', random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)

# Training
svm_model.fit(X_train_scaled, y_train)
knn_model.fit(X_train_scaled, y_train)

# Evaluasi
print(f"Akurasi SVM di test set: {svm_model.score(X_test_scaled, y_test):.4f}")
print(f"Akurasi KNN di test set: {knn_model.score(X_test_scaled, y_test):.4f}")

# Simpan model dan scaler
joblib.dump(scaler, 'model\scaler.pkl')
joblib.dump(svm_model, 'model\svm_model.pkl')
joblib.dump(knn_model, 'model\knn_model.pkl')

# Simpan data uji untuk evaluasi terpisah
np.savetxt('data_testing/fitur_testing.csv', X_test, delimiter=',')
np.savetxt('data_testing/label_testing.csv', y_test, delimiter=',')
print("✅ Model dan scaler berhasil disimpan.")
