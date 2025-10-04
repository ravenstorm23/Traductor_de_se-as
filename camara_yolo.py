import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from ultralytics import YOLO
from mediapipe import solutions as mp_solutions
from PIL import Image
import numpy as np

# -------------------------------
# Configuración
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLO
yolo_model_path = r"C:\Users\raven\Proyecto_de_profundizacion_traduccion_de_senas\yolo-hand-pose-main\model\best.pt"
yolo = YOLO(yolo_model_path)

# MediaPipe Hands
mp_hands = mp_solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ResNet18
resnet_model_path = r"C:\Users\raven\Proyecto_de_profundizacion_traduccion_de_senas\Traductor_de_senas\modelos_abecedario\mejor_modelo_resnet18.pth"
num_classes = 22
idx2letter = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'H',7:'I',8:'K',9:'L',10:'M',11:'N',12:'O',13:'P',14:'Q',15:'R',16:'T',17:'U',18:'V',19:'W',20:'X',21:'Y'}

resnet = models.resnet18(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
resnet.load_state_dict(torch.load(resnet_model_path, map_location=device))
resnet = resnet.to(device)
resnet.eval()

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# -------------------------------
# Función de predicción
# -------------------------------
def predict_letter(image):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = resnet(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    if conf.item() < 0.7:
        return "No reconocido"
    return idx2letter[pred.item()]

#---------------
# Recorte basado en landmarks
# -------------------------------
def crop_hand_with_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    x_coords = [lm.x * w for lm in landmarks.landmark]
    y_coords = [lm.y * h for lm in landmarks.landmark]

    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    # Añadir margen
    margin = 20
    x_min = max(x_min - margin, 0)
    y_min = max(y_min - margin, 0)
    x_max = min(x_max + margin, w)
    y_max = min(y_max + margin, h)

    return frame[y_min:y_max, x_min:x_max]

# -------------------------------
# Cámara en vivo
# -------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # YOLO detecta manos
    results = yolo(frame, conf=0.5)
    annotated_frame = results[0].plot()

    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            hand_crop = frame[y1:y2, x1:x2]

            letra = "No reconocido"
            if hand_crop.size > 0:
                hand_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
                hand_rgb.flags.writeable = False
                mp_result = hands_detector.process(hand_rgb)
                hand_rgb.flags.writeable = True

                if mp_result.multi_hand_landmarks:
                    landmarks = mp_result.multi_hand_landmarks[0]
                    refined_crop = crop_hand_with_landmarks(hand_crop, landmarks)
                    if refined_crop.size > 0:
                        letra = predict_letter(refined_crop)

            cv2.putText(annotated_frame, f"Letra: {letra}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0,255,0), 2)
            break  # solo primera mano

    else:
        cv2.putText(annotated_frame, "No reconocido", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("YOLO + MediaPipe + ResNet18", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
