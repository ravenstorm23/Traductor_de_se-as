import os
import shutil
import random
import time
import cv2
import numpy as np

# ========================
# CONFIG
# ========================
DATASET_DIR = r"C:\Users\raven\Proyecto_de_profundizacion_traduccion_de_senas\Traductor_de_senas\datasets"
EXCESS_DIR = r"C:\Users\raven\Proyecto_de_profundizacion_traduccion_de_senas\excedentes_datasets"
THRESHOLD = 10000

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

# ========================
# FUNCIONES AUXILIARES
# ========================
def is_image_file(name):
    return name.lower().endswith(IMG_EXTS)

def unique_move(src, dst_folder):
    """Mover archivo asegurando nombre único"""
    os.makedirs(dst_folder, exist_ok=True)
    base = os.path.basename(src)
    dst = os.path.join(dst_folder, base)
    if os.path.exists(dst):
        stamp = int(time.time() * 1000)
        name, ext = os.path.splitext(base)
        dst = os.path.join(dst_folder, f"{name}_moved{stamp}{ext}")
    shutil.move(src, dst)

def apply_modifications(img):
    """Devuelve una lista de versiones modificadas de la imagen"""
    mods = []

    # Escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mods.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    # Brillo aumentado
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
    mods.append(bright)

    # Oscurecido
    dark = cv2.convertScaleAbs(img, alpha=0.8, beta=-30)
    mods.append(dark)

    # Ruido gaussiano
    noise = np.random.normal(0, 25, img.shape).astype(np.int16)
    noisy = cv2.add(img.astype(np.int16), noise, dtype=cv2.CV_8U)
    mods.append(noisy)

    # Zoom (crop + resize)
    h, w = img.shape[:2]
    crop = img[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
    if crop.size > 0:
        zoom = cv2.resize(crop, (w, h))
        mods.append(zoom)

    return mods

def save_augmented(img, base_name, folder, count):
    """Guarda imagen modificada con nombre único"""
    name, ext = os.path.splitext(base_name)
    out_path = os.path.join(folder, f"{name}_aug{count}{ext}")
    cv2.imwrite(out_path, img)

# ========================
# PROCESAMIENTO
# ========================
def process_letra(folder):
    letra = os.path.basename(folder)
    files = [f for f in os.listdir(folder) if is_image_file(f)]
    total = len(files)

    # Caso 1: exceso de imágenes
    if total > THRESHOLD:
        exceso = total - THRESHOLD
        print(f"⚠️ {letra}: {total} imágenes, se moverán {exceso}")
        to_move = random.sample(files, exceso)
        for f in to_move:
            src = os.path.join(folder, f)
            dst_folder = os.path.join(EXCESS_DIR, letra)
            unique_move(src, dst_folder)
        print(f"📦 {letra}: movidas {exceso}, quedan {THRESHOLD}")
        return -exceso

    # Caso 2: falta de imágenes
    elif total < THRESHOLD:
        deficit = THRESHOLD - total
        print(f"➕ {letra}: {total} imágenes, se generarán {deficit}")
        i = 0
        while i < deficit:
            base_file = random.choice(files)
            base_path = os.path.join(folder, base_file)
            img = cv2.imread(base_path)

            if img is None:
                continue

            mods = apply_modifications(img)
            for m in mods:
                if i >= deficit:
                    break
                save_augmented(m, base_file, folder, int(time.time() * 1000 + i))
                i += 1

        print(f"✅ {letra}: ahora {THRESHOLD} imágenes (completado)")
        return deficit

    else:
        print(f"👌 {letra}: ya tiene {THRESHOLD} imágenes")
        return 0

def main():
    if not os.path.isdir(DATASET_DIR):
        print(f"❌ No existe dataset: {DATASET_DIR}")
        return
    os.makedirs(EXCESS_DIR, exist_ok=True)

    total_changed = 0
    letras = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    for letra in sorted(letras):
        changed = process_letra(os.path.join(DATASET_DIR, letra))
        total_changed += abs(changed)

    print("===================================")
    print(f"🏁 Proceso terminado, total archivos movidos/generados: {total_changed}")

if __name__ == "__main__":
    main()
