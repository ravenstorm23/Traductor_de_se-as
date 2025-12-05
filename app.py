from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from collections import deque, Counter
from PIL import Image
import time
import json
import os
import mediapipe as mp
import threading

# ==============================
# CONFIGURACIÓN
# ==============================
app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelos_abecedario", "mejor_modelo_resnet18.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_THRESHOLD = 0.80      # confianza mínima para considerar la predicción válida
STABLE_TIME = 2.5          # segundos que la letra debe mantenerse visibles (tiempo mínimo)
FRAMES_CONSISTENTES = 8    # número de frames consecutivos con misma predicción
NO_HAND_TIMEOUT = 3.5      # segundos sin detectar mano para guardar la palabra actual
CLEAR_TIMEOUT = 10.0       # ⏱ 10 segundos sin interacción para limpiar todo
DICCIONARIO_PATH = os.path.join(os.path.dirname(__file__), "diccionario.txt")

# Clases que el modelo reconoce (orden debe coincidir con el entrenamiento)
CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y'
]

# ==============================
# SERVICIO DE AUTOCOMPLETADO
# ==============================
class AutocompleteService:
    def __init__(self, dictionary_path):
        self.words = []
        self.load_dictionary(dictionary_path)

    def load_dictionary(self, path):
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.words = [line.strip().upper() for line in f if line.strip()]
                print(f" Diccionario cargado: {len(self.words)} palabras")
            else:
                print(" Diccionario no encontrado, usando lista básica")
                self.words = ["HOLA", "MUNDO", "COMO", "ESTAS", "GRACIAS", "ADIOS", "BUENOS", "DIAS"]
        except Exception as e:
            print(f" Error cargando diccionario: {e}")
            self.words = []

    def get_suggestions(self, prefix, limit=5):
        if not prefix:
            return []
        prefix = prefix.upper()
        suggestions = [w for w in self.words if w.startswith(prefix)]
        return suggestions[:limit]
    
    def is_valid_prefix(self, prefix):
        if not prefix: return True
        prefix = prefix.upper()
        # Retorna True si al menos una palabra empieza con este prefijo
        for w in self.words:
            if w.startswith(prefix):
                return True
        return False

autocomplete_service = AutocompleteService(DICCIONARIO_PATH)

# ==============================
# CARGA DEL MODELO RESNET
# ==============================
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASSES))
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f" Modelo ResNet cargado desde {MODEL_PATH}")
except Exception as e:
    print(f" Error cargando modelo ResNet: {e}")

model.to(DEVICE)
model.eval()

# ==============================
# TRANSFORMACIÓN DE IMAGEN
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==============================
# MEDIAPIPE HANDS
# ==============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5)

# ==============================
# VARIABLES DE ESTADO GLOBALES
# ==============================
class AppState:
    def __init__(self):
        self.ultima_prediccion = None
        self.tiempo_inicio_pred = 0
        self.frames_misma_letra = 0
        self.ultima_letra_confirmada = ""
        self.ultima_det_mano = time.time()
        self.tiempo_ultima_letra = time.time()
        self.letras_detectadas = []
        self.palabras = []
        
        # Estado para frontend
        self.current_letra = ""
        self.current_conf = 0.0
        self.hand_detected = False
        self.letra_estable_candidata = None # Para mostrar progreso visual
        
    def reset_all(self):
        self.ultima_prediccion = None
        self.tiempo_inicio_pred = 0
        self.frames_misma_letra = 0
        self.ultima_letra_confirmada = ""
        self.ultima_det_mano = time.time()
        self.tiempo_ultima_letra = time.time()
        self.letras_detectadas.clear()
        self.palabras.clear()
        self.current_letra = ""
        self.current_conf = 0.0
        self.hand_detected = False
        self.letra_estable_candidata = None
        print("🧹 Estado reiniciado")

    def finalizar_palabra(self):
        if self.letras_detectadas:
            palabra = ''.join(self.letras_detectadas)
            self.palabras.append(palabra)
            self.letras_detectadas.clear()
            self.ultima_letra_confirmada = ""
            return palabra
        return None

    def get_estado(self):
        # Calcular tiempo restante para estabilidad
        tiempo_restante = 0
        if self.letra_estable_candidata and self.tiempo_inicio_pred > 0:
            transcurrido = time.time() - self.tiempo_inicio_pred
            tiempo_restante = max(0, STABLE_TIME - transcurrido)
        
        # Obtener sugerencias de autocompletado
        palabra_actual = ''.join(self.letras_detectadas)
        sugerencias = autocomplete_service.get_suggestions(palabra_actual) if palabra_actual else []

        return {
            'letra': self.current_letra if (self.current_conf >= CONF_THRESHOLD and self.hand_detected) else "",
            'confianza': round(self.current_conf, 2),
            'letra_estable': self.letra_estable_candidata, # Candidata a ser confirmada
            'palabra_actual': palabra_actual,
            'frase': ' '.join(self.palabras),
            'letras': list(self.letras_detectadas),
            'tiempo_restante': round(tiempo_restante, 1),
            'hand_detected': self.hand_detected,
            'sugerencias': sugerencias
        }

state = AppState()
cap = None

# ==============================
# FUNCIÓN DE PREDICCIÓN (TOP K)
# ==============================
def predict_frame(frame, topk=3):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        # Obtener top K predicciones
        confs, preds = torch.topk(probs, topk, dim=1)
        
        results = []
        for i in range(topk):
            results.append((CLASSES[preds[0][i].item()], confs[0][i].item()))
        return results

# ==============================
# CLASE PARA LECTURA DE VIDEO EN HILO (NO BLOQUEANTE)
# ==============================
class CameraStream:
    def __init__(self, src=0):
        print(f" Inicializando cámara con source: {src}")
        # Usar DirectShow explícitamente en Windows para mejor compatibilidad
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        
        if not self.stream.isOpened():
            print("ERROR: No se pudo abrir la cámara")
            self.grabbed = False
            self.frame = None
            self.stopped = False
            self.lock = threading.Lock()
            return
        
        print("✅ Cámara abierta, configurando...")
        # Optimizaciones
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # ⏳ CRÍTICO: Dar tiempo a la cámara para inicializar
        print(" Esperando inicialización de hardware (2s)...")
        time.sleep(2.0)
        
        # Descartar primeros frames (pueden estar corruptos/vacíos)
        print("Descartando primeros frames...")
        for i in range(5):
            ret, frame = self.stream.read()
            if ret and frame is not None:
                print(f" Frame válido capturado ({i+1}/5)")
                self.grabbed = True
                self.frame = frame
                break
            time.sleep(0.2)
        else:
            print(" No se capturaron frames válidos, esperando al hilo...")
            self.grabbed = False
            self.frame = None
        
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        if self.stream.isOpened():
            print(" Iniciando hilo de captura")
        else:
            print("ERROR: Cámara no disponible")
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        print("Hilo de video iniciado...")
        frame_count = 0
        while True:
            if self.stopped:
                print(" Deteniendo hilo de video")
                return
            (grabbed, frame) = self.stream.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
            frame_count += 1
            if frame_count == 1:
                print(f" Primer frame capturado en hilo")

    def read(self):
        with self.lock:
            return self.frame.copy() if self.grabbed and self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.stream.release()
        print(" Cámara liberada")

# ==============================
# GENERADOR DE VIDEO
# ==============================
def generate_frames():
    global cap, state
    
    # URL de la cámara IP (si se usa celular) o 0 para webcam
    #CAMERA_SOURCE = "http://10.0.135.83:8080/video" 
    CAMERA_SOURCE = 0 
    
    # Usar la clase CameraStream para evitar lag
    if cap is None:
        cap = CameraStream(CAMERA_SOURCE).start()
        print(f" Cámara iniciada en hilo independiente: {CAMERA_SOURCE}")
    
    while True:
        frame = cap.read()
        if frame is None:
            # Si no hay frame, esperar un poco y reintentar
            time.sleep(0.1)
            # print(" Esperando frame...") # Descomentar para debug
            continue
            
        # Espejo SOLO si es webcam (0). Para IP Cam (celular trasero) no voltear.
        if CAMERA_SOURCE == 0:
            frame = cv2.flip(frame, 1)
        
        # Redimensionar si es muy grande (para mejorar rendimiento)
        h, w = frame.shape[:2]
        if w > 800:
            scale = 800 / w
            frame = cv2.resize(frame, (800, int(h * scale)))

            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        
        letra_predicha = ""
        conf = 0.0
        mano_detectada = False
        
        # ==============================
        # DETECCIÓN DE MANO (MEDIAPIPE)
        # ==============================
        if result.multi_hand_landmarks:
            mano_detectada = True
            state.ultima_det_mano = time.time()
            
            # Bounding box
            h, w, _ = frame.shape
            landmarks = result.multi_hand_landmarks[0].landmark
            x_min = int(min([lm.x for lm in landmarks]) * w)
            y_min = int(min([lm.y for lm in landmarks]) * h)
            x_max = int(max([lm.x for lm in landmarks]) * w)
            y_max = int(max([lm.y for lm in landmarks]) * h)
            
            # Añade margen para incluir toda la mano y algo de espacio
            margen = 40
            x_min = max(0, x_min - margen)
            y_min = max(0, y_min - margen)
            x_max = min(w, x_max + margen)
            y_max = min(h, y_max + margen)
            
            # Dibujar caja (Verde)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Extrae ROI y predice si el ROI no está vacío
            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size > 0:
                # Obtener top 3 predicciones
                top_preds = predict_frame(roi, topk=3)
                
                # Por defecto tomamos la mejor
                letra_predicha, conf = top_preds[0]
                
                # --- CORRECCIÓN POR CONTEXTO ---
                if state.letras_detectadas:
                    palabra_actual = "".join(state.letras_detectadas)
                    if not autocomplete_service.is_valid_prefix(palabra_actual + letra_predicha):
                        for alt_letra, alt_conf in top_preds[1:]:
                            if alt_conf > 0.4: 
                                if autocomplete_service.is_valid_prefix(palabra_actual + alt_letra):
                                    print(f" Corrección Contextual: {letra_predicha}({conf:.2f}) -> {alt_letra}({alt_conf:.2f})")
                                    letra_predicha = alt_letra
                                    conf = alt_conf
                                    break
                # -------------------------------
                
                state.tiempo_ultima_letra = time.time()
                
                # FILTRO Q
                if letra_predicha == 'Q':
                    letra_predicha = ""
                    conf = 0.0

        # Actualizar estado visual inmediato
        state.hand_detected = mano_detectada
        state.current_letra = letra_predicha
        state.current_conf = conf

        # ==============================
        # LÓGICA DE ESTABILIDAD
        # ==============================
        if mano_detectada and conf >= CONF_THRESHOLD and letra_predicha:
            if letra_predicha == state.ultima_prediccion:
                state.frames_misma_letra += 1
                state.letra_estable_candidata = letra_predicha # Para UI
                
                # Confirmar letra
                if state.frames_misma_letra >= FRAMES_CONSISTENTES and (time.time() - state.tiempo_inicio_pred >= STABLE_TIME):
                    if letra_predicha != state.ultima_letra_confirmada:
                        state.letras_detectadas.append(letra_predicha)
                        state.ultima_letra_confirmada = letra_predicha
                        print(f"🆕 Letra capturada: {letra_predicha}")
                        
                        # --- LÓGICA DE AUTO-COMPLETADO AUTOMÁTICO ---
                        if len(state.letras_detectadas) >= 3:
                            palabra_actual = ''.join(state.letras_detectadas)
                            sugerencias = autocomplete_service.get_suggestions(palabra_actual)
                            if len(sugerencias) == 1:
                                palabra_final = sugerencias[0]
                                print(f" Autocompletado automático: {palabra_actual} -> {palabra_final}")
                                state.letras_detectadas = list(palabra_final)
                                state.finalizar_palabra()
                        # --------------------------------------------
                    
                    # Reset parcial
                    state.frames_misma_letra = 0
                    state.ultima_prediccion = None
                    state.tiempo_inicio_pred = 0
                    state.letra_estable_candidata = None
            else:
                # Cambio de letra
                state.ultima_prediccion = letra_predicha
                state.tiempo_inicio_pred = time.time()
                state.frames_misma_letra = 1
                state.letra_estable_candidata = letra_predicha
        else:
            state.frames_misma_letra = 0
            state.letra_estable_candidata = None
            
        # ==============================
        # GESTIÓN DE TIEMPOS (Palabra y Limpieza)
        # ==============================
        if not mano_detectada:
            # Guardar palabra si pasa tiempo sin mano
            if (time.time() - state.ultima_det_mano) > NO_HAND_TIMEOUT and state.letras_detectadas:
                state.finalizar_palabra()
                print(" Palabra guardada automáticamente")
        
        # Limpieza total tras 10 segundos
        if (time.time() - state.tiempo_ultima_letra) > CLEAR_TIMEOUT and (state.palabras or state.letras_detectadas):
            state.reset_all()

        # NO DIBUJAR LANDMARKS (removido para limpieza visual)

        # Codificar frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==============================
# RUTAS FLASK
# ==============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_estado')
def get_estado():
    return jsonify(state.get_estado())

@app.route('/finalizar_palabra', methods=['POST'])
def finalizar_palabra_route():
    palabra = state.finalizar_palabra()
    return jsonify({'success': True, 'palabra': palabra, 'frase': ' '.join(state.palabras)})

@app.route('/limpiar', methods=['POST'])
def limpiar_route():
    state.reset_all()
    return jsonify({'success': True})

@app.route('/completar_palabra', methods=['POST'])
def completar_palabra():
    data = request.json
    palabra_completa = data.get('palabra')
    if palabra_completa:
        # Reemplazar las letras actuales con la palabra seleccionada
        state.letras_detectadas = list(palabra_completa)
        # Finalizar inmediatamente
        state.finalizar_palabra()
        return jsonify({'success': True})
    return jsonify({'success': False})

@app.route('/borrar_frase', methods=['POST'])
def borrar_frase():
    state.palabras.clear()
    return jsonify({'success': True})

@app.route('/hablar_frase', methods=['POST'])
def hablar_frase():
    """Endpoint para síntesis de voz de la frase completa"""
    try:
        import pyttsx3
        import threading
        
        frase = ' '.join(state.palabras)
        if not frase:
            return jsonify({'success': False, 'error': 'No hay frase para hablar'})
        
        # Ejecutar TTS en un hilo separado para no bloquear
        def speak():
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)  # Velocidad
                engine.setProperty('volume', 0.9)  # Volumen
                
                # Intentar voz en español
                voices = engine.getProperty('voices')
                for voice in voices:
                    if 'spanish' in voice.name.lower() or 'español' in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
                
                engine.say(frase)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"Error en TTS: {e}")
        
        thread = threading.Thread(target=speak, daemon=True)
        thread.start()
        
        return jsonify({'success': True, 'frase': frase})
    except ImportError:
        return jsonify({'success': False, 'error': 'pyttsx3 no está instalado'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print(" Servidor iniciado en http://localhost:5000")
    app.run(debug=True, threaded=True)
