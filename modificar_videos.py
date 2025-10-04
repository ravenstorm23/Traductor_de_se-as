import os
import shutil
import random
import cv2
import numpy as np
import time

# ========================
# CONFIG
# ========================
DATASET_DIR = r"C:\Users\raven\Proyecto_de_profundizacion_traduccion_de_senas\Traductor_de_senas\datasets_dinamicos"
EXCESS_DIR = r"C:\Users\raven\Proyecto_de_profundizacion_traduccion_de_senas\excedentes_videos"
TARGET = 1000
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

# ========================
# HELPERS
# ========================

def is_video_file(name):
    return name.lower().endswith(VIDEO_EXTS)

def unique_move(src, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    base = os.path.basename(src)
    dst = os.path.join(dst_folder, base)
    if os.path.exists(dst):
        stamp = int(time.time())
        name, ext = os.path.splitext(base)
        dst = os.path.join(dst_folder, f"{name}_moved{stamp}{ext}")
    shutil.move(src, dst)

def augment_video(src_path, dst_path, mode="gray"):
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise Exception(f"No se pudo abrir {src_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(dst_path, fourcc, fps, (width, height), True)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Transformaciones
        if mode == "gray":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif mode == "bright":
            frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=0)
        elif mode == "dark":
            frame = cv2.convertScaleAbs(frame, alpha=0.7, beta=0)
        elif mode == "mirror":
            frame = cv2.flip(frame, 1)
        frames.append(frame)

    # Si es "speed", reducimos frames para acelerar
    if mode == "speed":
        frames = frames[::2]  # 2x más rápido

    for f in frames:
        out.write(f)

    cap.release()
    out.release()

def process_letra(folder):
    letra = os.path.basename(folder)
    files = [f for f in os.listdir(folder) if is_video_file(f)]
    total = len(files)

    if total == TARGET:
        print(f"✅ {letra}: {total} videos (exacto)")
        return 0, 0
    elif total > TARGET:
        exceso = total - TARGET
        print(f"⚠️ {letra}: {total} videos, se moverán {exceso}")
        to_move = random.sample(files, exceso)
        for f in to_move:
            src = os.path.join(folder, f)
            dst_folder = os.path.join(EXCESS_DIR, letra)
            unique_move(src, dst_folder)
        return exceso, 0
    else:
        faltan = TARGET - total
        print(f"➕ {letra}: {total} videos, se crearán {faltan} nuevos")

        mods = ["gray", "bright", "dark", "mirror", "speed"]
        created = 0

        while created < faltan:
            src = os.path.join(folder, random.choice(files))
            name, ext = os.path.splitext(os.path.basename(src))
            mod = random.choice(mods)
            dst = os.path.join(folder, f"{name}_{mod}_{int(time.time())}{ext}")
            try:
                augment_video(src, dst, mode=mod)
                created += 1
            except Exception as e:
                print(f"❌ Error con {src}: {e}")
                continue

        return 0, faltan

def main():
    if not os.path.isdir(DATASET_DIR):
        print(f"❌ No existe dataset: {DATASET_DIR}")
        return
    os.makedirs(EXCESS_DIR, exist_ok=True)

    letras = ["G", "J", "S", "Z", "Ñ"]
    total_moved, total_created = 0, 0

    for letra in letras:
        folder = os.path.join(DATASET_DIR, letra)
        if not os.path.isdir(folder):
            print(f"⚠️ Carpeta no encontrada: {folder}")
            continue
        moved, created = process_letra(folder)
        total_moved += moved
        total_created += created

    print("===================================")
    print(f"📦 Total movidos: {total_moved}")
    print(f"🎬 Total creados: {total_created}")
    print("✅ Proceso terminado")

if __name__ == "__main__":
    main()
