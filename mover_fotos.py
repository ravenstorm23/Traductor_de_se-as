import os
import cv2
import shutil

# ========================
# CONFIG
# ========================
DATASET_DIR = r"C:\Users\raven\Proyecto_de_profundizacion_traduccion_de_senas\Traductor_de_senas\datasets"
OUTPUT_DIR = r"C:\Users\raven\Proyecto_de_profundizacion_traduccion_de_senas\datasets_pocas_fotos"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

MAX_IMGS = 1000  # límite por letra

# ========================
# FUNCIONES
# ========================
def is_image_file(name):
    return name.lower().endswith(IMG_EXTS)

def copiar_limitadas(letra_path, letra):
    files = [f for f in os.listdir(letra_path) if is_image_file(f)]
    files = sorted(files)[:MAX_IMGS]  # solo las primeras 1000

    dst_dir = os.path.join(OUTPUT_DIR, letra)
    os.makedirs(dst_dir, exist_ok=True)

    for f in files:
        src = os.path.join(letra_path, f)
        dst = os.path.join(dst_dir, f)
        shutil.copy(src, dst)

    print(f"📂 Letra {letra}: {len(files)} copiadas")

def main():
    if not os.path.isdir(DATASET_DIR):
        print(f"❌ No existe dataset: {DATASET_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    letras = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    for letra in sorted(letras):
        copiar_limitadas(os.path.join(DATASET_DIR, letra), letra)

    print("\n✅ Proceso completado. Dataset reducido en:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
