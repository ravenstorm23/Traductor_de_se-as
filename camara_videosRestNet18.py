import cv2  # OpenCV para captura y procesamiento de video
import torch  # PyTorch para deep learning
import torch.nn as nn  # Submódulo para redes neuronales
import torchvision.models as models  # Modelos preentrenados
import torchvision.transforms as transforms  # Transformaciones de imágenes
from collections import deque  # Para manejar secuencias de frames

# ========================
# CONFIG
# ========================
model_path = r"C:\Users\raven\Proyecto_de_profundizacion_traduccion_de_senas\Traductor_de_senas\modelos_abecedario\restnet_videos_lstm_best.pth"  # Ruta del modelo entrenado
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Usa GPU si está disponible

# Solo las letras de videos
classes = ["S", "J", "Z", "Ñ", "G"]  # Clases reconocidas por el modelo

# ========================
# MODELO
# ========================
class ResNetLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=1):
        super(ResNetLSTM, self).__init__()
        # Backbone ResNet18 -> Sequential (así se guardó tu modelo)
        backbone = models.resnet18(weights=None)  # Carga ResNet18 sin pesos preentrenados (solo arquitectura)
        self.resnet = nn.Sequential(*list(backbone.children())[:-1])  # Quita la última capa (solo convolucionales)

        self.lstm = nn.LSTM(
            input_size=512,  # Salida de ResNet18
            hidden_size=hidden_size,  # Tamaño del estado oculto
            num_layers=num_layers,  # Número de capas LSTM
            batch_first=True  # El batch es la primera dimensión
        )
        self.fc = nn.Linear(hidden_size, num_classes)  # Capa final para clasificación

    def forward(self, x):
        b, t, c, h, w = x.size()  # Tamaño del batch y secuencia
        x = x.view(b * t, c, h, w)  # Junta batch y secuencia para pasar por ResNet
        features = self.resnet(x)  # Extrae características (b*t, 512, 1, 1)
        features = features.view(b, t, -1)  # Vuelve a separar batch y secuencia (b, t, 512)
        lstm_out, _ = self.lstm(features)  # Pasa por LSTM
        out = self.fc(lstm_out[:, -1, :])  # Toma la salida del último frame
        return out

# Crear y cargar pesos
model = ResNetLSTM(num_classes=len(classes)).to(device)  # Instancia el modelo y lo mueve al dispositivo
state_dict = torch.load(model_path, map_location=device)  # Carga los pesos entrenados
model.load_state_dict(state_dict)  # Carga los pesos en el modelo
model.eval()  # Pone el modelo en modo evaluación

# ========================
# TRANSFORMACIONES
# ========================
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convierte el frame a imagen PIL
    transforms.Resize((224, 224)),  # Redimensiona la imagen
    transforms.ToTensor(),  # Convierte la imagen a tensor
    transforms.Normalize([0.485, 0.456, 0.406],  # Normaliza como en ImageNet
                         [0.229, 0.224, 0.225])
])

# ========================
# CÁMARA
# ========================
cap = cv2.VideoCapture(0)  # Inicia la captura de video desde la cámara
sequence = deque(maxlen=16)  # Buffer de frames para la secuencia

with torch.no_grad():  # No calcula gradientes (más eficiente)
    while True:
        ret, frame = cap.read()  # Lee un frame de la cámara
        if not ret:  # Si no se pudo leer, termina el bucle
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convierte el frame a RGB
        img_tensor = transform(frame_rgb).unsqueeze(0)  # Aplica transformaciones y añade dimensión batch
        sequence.append(img_tensor)  # Añade el frame al buffer

        if len(sequence) == 16:  # Cuando haya suficientes frames en el buffer
            seq_tensor = torch.stack(list(sequence), dim=1).to(device)  # (1, 16, 3, 224, 224)
            output = model(seq_tensor)  # Pasa la secuencia por el modelo
            _, pred = torch.max(output, 1)  # Obtiene la clase predicha
            label = classes[pred.item()]  # Traduce el índice a la letra
            cv2.putText(frame, f"Pred: {label}", (50, 50),  # Muestra la predicción en pantalla
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Sign Language Translator - Videos", frame)  # Muestra el frame en una ventana

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Si se presiona 'q', termina el bucle
            break

cap.release()  # Libera la cámara
cv2.destroyAllWindows()  # Cierra todas las ventanas de OpenCV
