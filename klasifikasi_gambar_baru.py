import sys
import numpy as np
import joblib
import pandas as pd
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from GUI.gui_klasifikasi_sampah import Ui_MainWindow

from feature_extraction_color import extract_color_features
from feature_extraction_shape import extract_shape_features
from feature_extraction_texture import extract_texture_features

label_map_inv = {0: 'Kertas', 1: 'Organik', 2: 'Plastik'}
color_map = {0: "#ff0000", 1: "#ff0000", 2: "#ff0000"}  # Biru, Hijau, Oranye

class ClassifyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.prediction_results = []

        # Tombol upload
        self.ui.pushButton.clicked.connect(self.upload_image)
        self.ui.pushButton_2.clicked.connect(self.resetapp)
        

        # Load model dan scaler
        self.scaler = joblib.load('model\scaler.pkl')
        self.svm_model = joblib.load('model\svm_model.pkl')
        self.knn_model = joblib.load('model\knn_model.pkl')
 

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar Sampah", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.show_image(file_path)
            self.classify_image(file_path)

    def show_image(self, path):
        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(self.ui.label_4.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.ui.label_4.setPixmap(pixmap)
        self.ui.label_4.setStyleSheet("border: none;")

    def classify_image(self, path):

            # Ekstraksi fitur
            color_feat = extract_color_features(path)
            shape_feat = extract_shape_features(path)
            texture_feat = extract_texture_features(path)

            combined_feat = np.concatenate([color_feat, shape_feat, texture_feat]).reshape(1, -1)
            scaled_feat = self.scaler.transform(combined_feat)

            # Prediksi
            svm_pred = self.svm_model.predict(scaled_feat)[0]
            knn_pred = self.knn_model.predict(scaled_feat)[0]

            svm_color = color_map.get(svm_pred, "#000")
            knn_color = color_map.get(knn_pred, "#000")

            result_html = (
                f"<b>SVM memprediksi:</b> <span style='color:{svm_color}'>{label_map_inv[svm_pred]}</span><br>"
                f"<b>KNN memprediksi:</b> <span style='color:{knn_color}'>{label_map_inv[knn_pred]}</span>"
            )

            self.ui.label_5.setText(result_html)

             # Simpan hasil prediksi ke CSV dengan label string
            self.prediction_results.append({
                "nama_file": path.split("/")[-1],
                "SVM": label_map_inv[svm_pred],
                "KNN": label_map_inv[knn_pred]
            })

            print("✅ Hasil prediksi disimpan ke 'hasil_prediksi.csv'")


    def closeEvent(self, event):
        if self.prediction_results:
            df = pd.DataFrame(self.prediction_results)
            df.to_csv("hasil_prediksi.csv", index=False)
            print("✅ Semua hasil prediksi disimpan ke 'hasil_prediksi.csv'")
        else:
            print("⚠️ Tidak ada hasil yang disimpan.")
        event.accept()  # penting! agar jendela tetap bisa ditutup





    def resetapp(self):
        self.ui.label_4.clear()
        self.ui.label_4.setText("Belum ada gamba yang dipilih")
        self.ui.label_4.setAlignment(Qt.AlignCenter)
        self.ui.label_4.setStyleSheet("border: 1px dashed #aaa; background-color: white; color: #000000;")
        self.ui.label_5.clear()

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClassifyApp()
    window.setWindowTitle("Klasifikasi Sampah - Sri dan Ilham")
    window.show()
    sys.exit(app.exec_())
