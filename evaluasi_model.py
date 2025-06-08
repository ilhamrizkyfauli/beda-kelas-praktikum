import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load model dan scaler
scaler = joblib.load('model/scaler.pkl')
svm_model = joblib.load('model/svm_model.pkl')
knn_model = joblib.load('model/knn_model.pkl')

# Load data uji
X_test = np.loadtxt('data_testing/fitur_testing.csv', delimiter=',')
y_test = np.loadtxt('data_testing/label_testing.csv', delimiter=',')

# Transformasi dengan scaler
X_test_scaled = scaler.transform(X_test)

# Evaluasi SVM
svm_preds = svm_model.predict(X_test_scaled)
print("====================== Evaluasi SVM =======================")
print(classification_report(y_test, svm_preds, digits=4))
print("Confusion Matrix SVM:")
print(confusion_matrix(y_test, svm_preds))
print(f"Akurasi SVM: {accuracy_score(y_test, svm_preds):.4f}")

# Evaluasi KNN
knn_preds = knn_model.predict(X_test_scaled)
print("\n===================== Evaluasi KNN =====================")
print(classification_report(y_test, knn_preds, digits=4))
print("Confusion Matrix KNN:")
print(confusion_matrix(y_test, knn_preds))
print(f"Akurasi KNN: {accuracy_score(y_test, knn_preds):.4f}")
