import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import time
from collections import deque, Counter
import mediapipe as mp
import numpy as np   # ✅ SOLUCIÓN: importar numpy


# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ruta del modelo
model_path = r"C:\Users\raven\Proyecto_de_profundizacion_traduccion_de_senas\Traductor_de_senas\modelos_abecedario\mejor_modelo_resnet18.pth"
num_classes = 22  

# Diccionario índice → letra
idx2letter = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'H', 7:'I',
    8:'K', 9:'L', 10:'M', 11:'N', 12:'O', 13:'P', 14:'Q',
    15:'R', 16:'T', 17:'U', 18:'V', 19:'W', 20:'X', 21:'Y'
}

# Transformaciones para imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Carga modelo
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

def segment_hand(frame):
    """Detecta la mano y devuelve la región segmentada"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask[:] = 0  # máscara en negro

    if result.multi_hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in result.multi_hand_landmarks:
            points = []
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                points.append((x, y))
            hull = cv2.convexHull(np.array(points))
            cv2.fillConvexPoly(mask, hull, 255)

    segmented = cv2.bitwise_and(frame, frame, mask=mask)
    return segmented

def predict_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return idx2letter[predicted.item()]

# Captura de video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

fps_interval = 0.1
last_time = time.time()
buffer_size = 5
pred_buffer = deque(maxlen=buffer_size)
current_display = ''

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_original = cv2.flip(frame, 1)
    
    # ✅ Segmentación de la mano
    segmented_hand = segment_hand(frame_original)

    current_time = time.time()
    if current_time - last_time >= fps_interval:
        immediate_pred = predict_frame(segmented_hand)
        pred_buffer.append(immediate_pred)
        last_time = current_time

        most_common = Counter(pred_buffer).most_common(1)[0][0]
        if immediate_pred == most_common:
            current_display = immediate_pred
        else:
            current_display = most_common

    cv2.putText(frame_original, f"Letra: {current_display}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # Muestra original y mano segmentada en paralelo
    cv2.imshow("Traductor de Señas - ResNet18", frame_original)
    cv2.imshow("Mano Segmentada", segmented_hand)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
