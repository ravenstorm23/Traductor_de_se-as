import cv2
import os
import shutil
import numpy as np
import mediapipe as mp

# Detector de manos con MediaPipe
mp_hands = mp.solutions.hands

# ====== AUGMENTACIONES ROBUSTAS ======
def aplicar_gris(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir {video_path}")
        return False

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gris_rgb = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
        out.write(gris_rgb)
        frame_count += 1

    cap.release()
    out.release()
    if frame_count == 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    return True

def aplicar_brillo(video_path, output_path, factor=50):
    return aplicar_modificacion(video_path, output_path, lambda f: cv2.convertScaleAbs(f, alpha=1, beta=factor))

def aplicar_oscuro(video_path, output_path, factor=-50):
    return aplicar_modificacion(video_path, output_path, lambda f: cv2.convertScaleAbs(f, alpha=1, beta=factor))

def aplicar_ruido(video_path, output_path):
    def ruido(f):
        ruido = np.random.normal(0, 25, f.shape).astype(np.uint8)
        return cv2.add(f, ruido)
    return aplicar_modificacion(video_path, output_path, ruido)

def aplicar_zoom(video_path, output_path, zoom_factor=1.2):
    def zoom(f):
        h, w = f.shape[:2]
        nh, nw = int(h / zoom_factor), int(w / zoom_factor)
        y1, x1 = (h - nh) // 2, (w - nw) // 2
        recorte = f[y1:y1+nh, x1:x1+nw]
        return cv2.resize(recorte, (w, h))
    return aplicar_modificacion(video_path, output_path, zoom)

def aplicar_modificacion(video_path, output_path, funcion):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir {video_path}")
        return False

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        modificado = funcion(frame)
        out.write(modificado)
        frame_count += 1

    cap.release()
    out.release()
    if frame_count == 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    return True


# ====== VERIFICACIÓN DE VIDEOS ======
def evaluar_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = 0
    frames_con_landmarks = 0

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                frames_con_landmarks += 1

    cap.release()
    if total_frames == 0:
        return 0.0
    return (frames_con_landmarks / total_frames) * 100


# ====== PIPELINE SOLO PARA Ñ ======
def procesar_dataset(dataset_dir="datasets_dinamicos", fallidos_dir="videos_fallidos", threshold=10.0):
    sufijos_modificaciones = ["_gris", "_brillo", "_oscuro", "_ruido", "_zoom"]

    if not os.path.exists(fallidos_dir):
        os.makedirs(fallidos_dir)

    letra = "Ñ"
    letra_path = os.path.join(dataset_dir, letra)
    if not os.path.isdir(letra_path):
        print(f"⚠️ No existe carpeta para la letra {letra}")
        return

    destino_carpeta = os.path.join(fallidos_dir, letra)
    os.makedirs(destino_carpeta, exist_ok=True)

    for video_file in os.listdir(letra_path):
        video_path = os.path.join(letra_path, video_file)

        if not os.path.exists(video_path):
            print(f"⚠️ Archivo no encontrado, saltando: {video_path}")
            continue

        # Si ya está modificado, solo verificar
        if any(sufijo in video_file for sufijo in sufijos_modificaciones):
            porcentaje = evaluar_video(video_path)
            if porcentaje < threshold:
                print(f"❌ {video_file} rechazado ({porcentaje:.2f}%), moviendo a fallidos")
                try:
                    shutil.move(video_path, os.path.join(destino_carpeta, video_file))
                except Exception as e:
                    print(f"⚠️ Error al mover {video_file}: {e}")
            else:
                print(f"🔎 {video_file} (modificado) verificado ✅ ({porcentaje:.2f}%)")
            continue

        # Si es original → generar augmentaciones
        base, ext = os.path.splitext(video_file)
        modificaciones = {
            "_gris": aplicar_gris,
            "_brillo": aplicar_brillo,
            "_oscuro": aplicar_oscuro,
            "_ruido": aplicar_ruido,
            "_zoom": aplicar_zoom
        }

        for sufijo, funcion in modificaciones.items():
            output_path = os.path.join(letra_path, f"{base}{sufijo}{ext}")
            if not os.path.exists(output_path):  # no rehacer si ya existe
                exito = funcion(video_path, output_path)
                if exito:
                    print(f"✅ Augmentación {sufijo} creada: {output_path}")
                else:
                    print(f"⚠️ Augmentación {sufijo} fallida para {video_file}")

        # Verificar original también
        porcentaje = evaluar_video(video_path)
        if porcentaje < threshold:
            print(f"❌ {video_file} (original) rechazado ({porcentaje:.2f}%), moviendo a fallidos")
            try:
                shutil.move(video_path, os.path.join(destino_carpeta, video_file))
            except Exception as e:
                print(f"⚠️ Error al mover {video_file}: {e}")
        else:
            print(f"✅ {video_file} (original) aceptado ({porcentaje:.2f}%)")


if __name__ == "__main__":
    procesar_dataset()
