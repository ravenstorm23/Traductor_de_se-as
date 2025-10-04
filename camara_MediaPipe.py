import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import mediapipe as mp
import numpy as np
from PIL import Image
import time
from collections import deque, Counter
import os

# ================================
# CONFIGURACIÓN
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = r"C:\Users\raven\Proyecto_de_profundizacion_traduccion_de_senas\Traductor_de_senas\modelos_abecedario\mejor_modelo_resnet18.pth"
num_classes = 22

idx2letter = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'H', 7:'I',
    8:'K', 9:'L', 10:'M', 11:'N', 12:'O', 13:'P', 14:'Q',
    15:'R', 16:'T', 17:'U', 18:'V', 19:'W', 20:'X', 21:'Y'
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ================================
# MEDIAPIPE
# ================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# ================================
# MEJORA DE ILUMINACIÓN
# ================================
def mejorar_iluminacion(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    hsv_mejorado = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_mejorado, cv2.COLOR_HSV2BGR)

# ================================
# SEGMENTACIÓN DE LA MANO
# ================================
def segment_hand(frame, expand_ratio=1.5):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if result.multi_hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in result.multi_hand_landmarks:
            points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark])

            center = np.mean(points, axis=0)
            expanded_points = center + (points - center) * expand_ratio
            expanded_points = np.clip(expanded_points, [0, 0], [w - 1, h - 1]).astype(np.int32)

            hull = cv2.convexHull(expanded_points)
            cv2.fillConvexPoly(mask, hull, 255)

    segmented = cv2.bitwise_and(frame, frame, mask=mask)
    return segmented

# ================================
# PREDICCIÓN
# ================================
def predict_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    if conf.item() < 0.7:
        return None
    return idx2letter[pred.item()]

# ================================
# LOOP DE CÁMARA
# ================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

fps_interval = 0.1
last_time = time.time()
buffer_size = 5
pred_buffer = deque(maxlen=buffer_size)
current_display = ''
expand_ratio = 1.5

save_dir = "segmentados_guardados"
os.makedirs(save_dir, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_original = cv2.flip(frame, 1)
    frame_iluminado = mejorar_iluminacion(frame_original)

    segmented_hand = segment_hand(frame_iluminado, expand_ratio)

    current_time = time.time()
    if current_time - last_time >= fps_interval:
        immediate_pred = predict_frame(segmented_hand)
        if immediate_pred:
            pred_buffer.append(immediate_pred)
            most_common = Counter(pred_buffer).most_common(1)[0][0]
            current_display = most_common
        else:
            current_display = ''

        last_time = current_time

    if current_display:
        cv2.putText(frame_original, f"Letra: {current_display}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Traductor de Señas - ResNet18", frame_original)
    cv2.imshow("Mano Segmentada", segmented_hand)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+'):
        expand_ratio += 0.1
        print(f"🔧 expand_ratio aumentado a: {expand_ratio:.2f}")
    elif key == ord('-'):
        expand_ratio = max(1.0, expand_ratio - 0.1)
        print(f"🔧 expand_ratio reducido a: {expand_ratio:.2f}")
    elif key == ord('s') and current_display:
        filename = f"{save_dir}/{current_display}_{int(current_time)}.jpg"
        cv2.imwrite(filename, segmented_hand)
        print(f"💾 Imagen guardada: {filename}")

cap.release()
cv2.destroyAllWindows()
hands.close()
