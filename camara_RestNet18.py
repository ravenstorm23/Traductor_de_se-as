import torch  # Importa PyTorch para deep learning
import torch.nn as nn  # Submódulo para redes neuronales
from torchvision import models, transforms  # Modelos y transformaciones de imágenes
import cv2  # OpenCV para captura y procesamiento de video
from PIL import Image  # Para manipulación de imágenes
import time  # Para manejo de tiempos y FPS
from collections import deque, Counter  # Para suavizado de predicciones

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Usa GPU si está disponible, si no CPU
model_path = r"C:\Users\raven\Proyecto_de_profundizacion_traduccion_de_senas\Traductor_de_senas\modelos_abecedario\mejor_modelo_resnet18.pth"
num_classes = 22  # Número de clases (letras reconocidas)

# Diccionario para convertir índice de clase a letra
idx2letter = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'H', 7:'I',
    8:'K', 9:'L', 10:'M', 11:'N', 12:'O', 13:'P', 14:'Q',
    15:'R', 16:'T', 17:'U', 18:'V', 19:'W', 20:'X', 21:'Y'
}

# Transformaciones para la imagen antes de pasarla al modelo
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensiona la imagen a 224x224
    transforms.ToTensor(),  # Convierte la imagen a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normaliza con media y std de ImageNet
                         std=[0.229, 0.224, 0.225])
])

# Carga el modelo ResNet18 y ajusta la última capa para el número de clases
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Ajusta la capa final
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))  # Carga los pesos entrenados
model = model.to(device)  # Mueve el modelo al dispositivo
model.eval()  # Pone el modelo en modo evaluación

def predict_frame(frame):
    # Convierte el frame de OpenCV a PIL, aplica transformaciones y predice la letra
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convierte BGR a RGB y a PIL
    image = transform(image).unsqueeze(0).to(device)  # Aplica transformaciones y añade dimensión batch
    with torch.no_grad():  # No calcula gradientes
        outputs = model(image)  # Pasa la imagen por el modelo
        _, predicted = torch.max(outputs, 1)  # Obtiene la clase predicha
    return idx2letter[predicted.item()]  # Devuelve la letra correspondiente

cap = cv2.VideoCapture(0)  # Inicia la captura de video desde la cámara
if not cap.isOpened():  # Verifica si la cámara se abrió correctamente
    print("No se pudo abrir la cámara")
    exit()

fps_interval = 0.1  # Intervalo de tiempo entre predicciones (en segundos)
last_time = time.time()  # Guarda el tiempo de la última predicción
buffer_size = 5  # Tamaño del buffer para suavizar predicciones
pred_buffer = deque(maxlen=buffer_size)  # Buffer circular para predicciones recientes
current_display = ''  # Letra que se muestra actualmente

while True:  # Bucle principal de captura y predicción
    ret, frame = cap.read()  # Lee un frame de la cámara
    if not ret:  # Si no se pudo leer, termina el bucle
        break

    frame_original = cv2.flip(frame, 1)  # Invierte horizontalmente la imagen para efecto espejo
    frame_model = frame_original.copy()  # Copia para procesar y pasar al modelo

    # Preprocesamiento: mejora el contraste y convierte a escala de grises
    hsv = cv2.cvtColor(frame_model, cv2.COLOR_BGR2HSV)  # Convierte a HSV
    h, s, v = cv2.split(hsv)  # Separa canales
    v = cv2.equalizeHist(v)  # Ecualiza el canal de brillo
    hsv = cv2.merge([h, s, v])  # Une los canales
    frame_model = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # Vuelve a BGR
    gray = cv2.cvtColor(frame_model, cv2.COLOR_BGR2GRAY)  # Convierte a escala de grises
    frame_model = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Vuelve a BGR para el modelo

    current_time = time.time()  # Tiempo actual
    if current_time - last_time >= fps_interval:  # Si pasó el intervalo de tiempo
        immediate_pred = predict_frame(frame_model)  # Predice la letra del frame
        pred_buffer.append(immediate_pred)  # Añade la predicción al buffer
        last_time = current_time  # Actualiza el tiempo

        # Suavizado con buffer: obtiene la predicción más común en el buffer
        most_common = Counter(pred_buffer).most_common(1)[0][0]

        # Muestra la predicción inmediata si coincide con la mayoría, si no mantiene la del buffer
        if immediate_pred == most_common:
            current_display = immediate_pred
        else:
            current_display = most_common

    # Muestra la letra predicha en la ventana
    cv2.putText(frame_original, f"Letra: {current_display}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Traductor de Señas - ResNet18", frame_original)  # Muestra el frame en una ventana

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Si se presiona 'q', termina el bucle
        break

cap.release()  # Libera la cámara
cv2.destroyAllWindows()  # Cierra todas las ventanas de OpenCV
