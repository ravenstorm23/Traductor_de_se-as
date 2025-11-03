import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import mediapipe as mp
import time

# ==============================
# CONFIGURACIÓN
# ==============================
# Ruta al modelo entrenado y dispositivo (GPU si está disponible)
MODEL_PATH = r"C:\Users\Pc\Desktop\Traductor_de_senas\Traductor_de_senas\modelos_abecedario\mejor_modelo_resnet18.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Umbrales y tiempos para la lógica de captura automática
CONF_THRESHOLD = 0.90      # confianza mínima para considerar la predicción válida
STABLE_TIME = 2.5          # segundos que la letra debe mantenerse visibles (tiempo mínimo)
FRAMES_CONSISTENTES = 8    # número de frames consecutivos con misma predicción
NO_HAND_TIMEOUT = 3.5      # segundos sin detectar mano para guardar la palabra actual
CLEAR_TIMEOUT = 10.0       # segundos sin interacción para limpiar todo

# Clases que el modelo reconoce (orden debe coincidir con el entrenamiento)
CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y'
]

# ==============================
# CARGA DEL MODELO
# ==============================
# Construye ResNet18 y reemplaza la última capa según el número de clases
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASSES))
# Carga pesos guardados (map_location asegura compatibilidad CPU/GPU)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()  # modo evaluación: desactiva dropout, batchnorm en training, etc.

# ==============================
# TRANSFORMACIÓN DE IMAGEN
# ==============================
# Transformaciones idénticas a las usadas en entrenamiento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==============================
# MEDIAPIPE HANDS (detección de mano)
# ==============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Configuración de MediaPipe: no es modo estático y máximo 1 mano
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5)

# ==============================
# VARIABLES DE ESTADO (para la lógica)
# ==============================
ultima_prediccion = None      # última predicción detectada (cadena)
tiempo_inicio_pred = 0        # tiempo en que empezó la observación de la predicción actual
frames_misma_letra = 0        # contador de frames consecutivos con misma predicción
ultima_letra_confirmada = ""  # última letra que se añadió definitivamente (evita duplicados)
ultima_det_mano = time.time() # instante de la última detección de mano
tiempo_ultima_letra = time.time()  # tiempo desde la última letra registrada
letras_detectadas = []        # lista de letras de la palabra en construcción
palabras = []                 # lista de palabras ya guardadas

# ==============================
# FUNCIÓN DE PREDICCIÓN
# ==============================
def predict_frame(frame):
    """
    Toma un ROI (BGR, como viene de OpenCV), lo convierte a PIL RGB,
    aplica transformaciones y pasa por el modelo para obtener clase y confianza.
    Devuelve (letra, confianza).
    """
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        return CLASSES[pred.item()], conf.item()

# ==============================
# CAPTURA DE CÁMARA
# ==============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

print("✅ Traductor de señas automático iniciado.")
print("Controles: [C] = limpiar todo | [Supr] = borrar última letra | [ESC] = salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Voltea la imagen para que se vea como un espejo (más natural para el usuario)
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Procesa con MediaPipe para detectar manos y sus landmarks
    result = hands.process(rgb_frame)

    letra_predicha = ""
    conf = 0.0
    mano_detectada = False

    # ==============================
    # DETECCIÓN DE MANO Y EXTRACCIÓN DE ROI
    # ==============================
    if result.multi_hand_landmarks:
        mano_detectada = True
        ultima_det_mano = time.time()  # actualiza tiempo de última detección de mano

        # Calcula bounding box a partir de landmarks normalizados (0..1) -> píxeles
        h, w, _ = frame.shape
        x_min = int(min([lm.x for lm in result.multi_hand_landmarks[0].landmark]) * w)
        y_min = int(min([lm.y for lm in result.multi_hand_landmarks[0].landmark]) * h)
        x_max = int(max([lm.x for lm in result.multi_hand_landmarks[0].landmark]) * w)
        y_max = int(max([lm.y for lm in result.multi_hand_landmarks[0].landmark]) * h)

        # Añade margen para incluir toda la mano y algo de espacio
        margen = 40
        x_min = max(0, x_min - margen)
        y_min = max(0, y_min - margen)
        x_max = min(w, x_max + margen)
        y_max = min(h, y_max + margen)

        # Extrae ROI y predice si el ROI no está vacío
        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size > 0:
            letra_predicha, conf = predict_frame(roi)
            tiempo_ultima_letra = time.time()  # reinicia contador para limpieza automática

    # ==============================
    # LÓGICA DE CAPTURA AUTOMÁTICA (estabilidad y umbrales)
    # ==============================
    # Solo procesa si hay mano y la confianza supera el umbral
    if mano_detectada and conf >= CONF_THRESHOLD:
        if letra_predicha == ultima_prediccion:
            frames_misma_letra += 1
            # Si hay suficiente repetición en frames y tiempo estable, confirma letra
            if frames_misma_letra >= FRAMES_CONSISTENTES and time.time() - tiempo_inicio_pred >= STABLE_TIME:
                if letra_predicha != ultima_letra_confirmada:  # evita añadir la misma letra repetida
                    letras_detectadas.append(letra_predicha)
                    ultima_letra_confirmada = letra_predicha
                    print(f"🆕 Letra capturada: {letra_predicha}")
                # reinicia contadores para la siguiente letra
                frames_misma_letra = 0
                ultima_prediccion = None
                tiempo_inicio_pred = 0
        else:
            # nueva predicción diferente: empieza a contar estabilidad
            ultima_prediccion = letra_predicha
            tiempo_inicio_pred = time.time()
            frames_misma_letra = 1
    else:
        # si no hay mano o confianza baja: resetea contador de frames consecutivas
        frames_misma_letra = 0

    # ==============================
    # GESTIÓN DE TIEMPOS SIN MANO (guardar palabras y limpieza)
    # ==============================
    if not mano_detectada:
        # Si ha pasado NO_HAND_TIMEOUT sin mano y hay letras acumuladas, guarda palabra
        if (time.time() - ultima_det_mano) > NO_HAND_TIMEOUT and letras_detectadas:
            palabra = ''.join(letras_detectadas)
            palabras.append(palabra)
            letras_detectadas.clear()
            ultima_letra_confirmada = ""
            print(f"✅ Palabra guardada automáticamente: {palabra}")

        # Si ha pasado CLEAR_TIMEOUT desde la última letra, limpia todo para reset
        if (time.time() - tiempo_ultima_letra) > CLEAR_TIMEOUT and (palabras or letras_detectadas):
            print("⚠️ 10 segundos sin detección. Limpiando todo.")
            letras_detectadas.clear()
            palabras.clear()
            ultima_prediccion = None
            frames_misma_letra = 0
            ultima_letra_confirmada = ""
            tiempo_inicio_pred = 0
            tiempo_ultima_letra = time.time()

    # ==============================
    # VISUALIZACIÓN EN PANTALLA
    # ==============================
    color = (0, 255, 0) if mano_detectada else (0, 0, 255)
    palabra_actual = ''.join(letras_detectadas)
    frase = ' '.join(palabras) if palabras else ""

    # Muestra la letra y su confianza (o indica que no hay mano)
    texto_letra = f"Letra: {letra_predicha} ({conf:.2f})" if mano_detectada else "Letra: [sin mano]"
    cv2.putText(frame, texto_letra, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Palabra: {palabra_actual}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Frase: {frase}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Dibuja landmarks de la mano si fue detectada
    if mano_detectada:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Traductor Automático de Señas", frame)

    # ==============================
    # CONTROLES DEL TECLADO
    # ==============================
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC -> salir
        break
    elif key in [ord('c'), ord('C')]:  # limpiar todo manualmente
        letras_detectadas.clear()
        palabras.clear()
        ultima_letra_confirmada = ""
        print("🧹 Limpieza manual ejecutada")
    elif key == 46:  # Tecla Supr/Delete -> borrar última letra detectada
        if letras_detectadas:
            letra_eliminada = letras_detectadas.pop()
            print(f"❌ Letra eliminada: {letra_eliminada}")
        else:
            print("⚠️ No hay letras para eliminar")

cap.release()
cv2.destroyAllWindows()

# ==============================
# RESULTADO FINAL (al cerrar la app)
# ==============================
if palabras or letras_detectadas:
    if letras_detectadas:
        palabras.append(''.join(letras_detectadas))
    print("\n" + "="*50)
    print("📝 FRASE FINAL:")
    print(' '.join(palabras))
    print("="*50)
