import sys
import cv2
import numpy as np
import joblib
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFrame, QStackedLayout
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
import mediapipe as mp

class SignLanguageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signa - Traductor de Señas")
        self.setFixedSize(400, 700)
        self.setStyleSheet("background-color: #121212;")

        # Stack para pantalla de bienvenida y main app
        self.stack = QStackedLayout()
        self.setLayout(self.stack)

        # --- Pantalla de bienvenida con nombre "Signa" ---
        self.splash = QLabel("Signa")
        self.splash.setAlignment(Qt.AlignCenter)
        self.splash.setFont(QFont("Arial", 60, QFont.Bold))
        self.splash.setStyleSheet("color: #00ffcc; background-color: #121212;")
        self.stack.addWidget(self.splash)

        # --- Pantalla principal ---
        self.main_widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Título
        self.title_label = QLabel("Signa")
        self.title_label.setFont(QFont("Arial", 22, QFont.Bold))
        self.title_label.setStyleSheet("color: #00ffcc;")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        # Marco para cámara
        self.camera_frame = QFrame()
        self.camera_frame.setFixedSize(360, 400)
        self.camera_frame.setStyleSheet("""
            background-color: #1e1e1e;
            border-radius: 20px;
            border: 2px solid #00d4cc;
        """)
        self.camera_label = QLabel(self.camera_frame)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setGeometry(0, 0, 360, 400)
        layout.addWidget(self.camera_frame, alignment=Qt.AlignCenter)

        # Label predicción
        self.prediction_label = QLabel("Predicción: ---")
        self.prediction_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.prediction_label.setStyleSheet("color: #00ffcc;")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prediction_label)

        # Botón iniciar
        self.start_btn = QPushButton("Iniciar Cámara")
        self.start_btn.setFont(QFont("Arial", 16, QFont.Bold))
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4cc;
                color: #121212;
                border-radius: 15px;
                padding: 15px;
            }
            QPushButton:hover {
                background-color: #00ffc9;
            }
        """)
        self.start_btn.clicked.connect(self.start_camera)
        layout.addWidget(self.start_btn, alignment=Qt.AlignCenter)

        # Botón detener
        self.stop_btn = QPushButton("Detener Cámara")
        self.stop_btn.setFont(QFont("Arial", 16, QFont.Bold))
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #00aaff;
                color: #121212;
                border-radius: 15px;
                padding: 15px;
            }
            QPushButton:hover {
                background-color: #0099dd;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_camera)
        layout.addWidget(self.stop_btn, alignment=Qt.AlignCenter)

        self.main_widget.setLayout(layout)
        self.stack.addWidget(self.main_widget)

        # Variables
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1)
        self.mp_drawing = mp.solutions.drawing_utils

        # Cargar modelo
        data = joblib.load(
            "C:\\Users\\raven\\Proyecto_de_profundizacion_traduccion_de_senas\\Traductor_de_senas\\modelos_abecedario\\modelo_abecedario_mediapipe.joblib"
        )
        self.model = data['modelo']
        self.scaler = data.get('scaler', None)

        # Mostrar splash 2 segundos antes de ir a main app
        QTimer.singleShot(2000, lambda: self.stack.setCurrentWidget(self.main_widget))

    def find_working_camera(self, max_index=5):
        for i in range(max_index):
            temp_cap = cv2.VideoCapture(i)
            if temp_cap.isOpened():
                temp_cap.release()
                return i
        return None

    def start_camera(self):
        cam_index = self.find_working_camera()
        if cam_index is None:
            self.prediction_label.setText("No se detectó ninguna cámara")
            return
        self.capture = cv2.VideoCapture(cam_index)
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        if self.capture:
            self.capture.release()
        self.camera_label.clear()

    def update_frame(self):
        if self.capture is None:
            return
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Espejar tipo selfie
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                X = np.array([coords])
                if self.scaler:
                    X = self.scaler.transform(X)
                prediction_letter = self.model.predict(X)[0]
                self.prediction_label.setText(f"Predicción: {prediction_letter}")
            else:
                self.prediction_label.setText("Predicción: ---")

            # Mostrar frame
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        if self.capture:
            self.capture.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())
