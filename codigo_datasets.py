# CONFIGURACIÓN CON MEDIAPIPE
import os
import cv2
import numpy as np
import pandas as pd
import joblib
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import mediapipe as mp

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# RUTAS
PROYECTO_PATH = r'C:\Users\raven\Proyecto_de_profundizacion_traduccion _de_senas\Traductor_de_senas'
DATASET_PATH = os.path.join(PROYECTO_PATH, 'datasets')
DATA_DIR = os.path.join(PROYECTO_PATH, 'data_lsc_abecedario')
MODEL_DIR = os.path.join(PROYECTO_PATH, 'modelos_abecedario')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ABECEDARIO
LETRAS = list("abcdefghijklmnopqrstuvwxyz")

print(f"📁 Dataset: {DATASET_PATH}")
print(f"🔤 Letras: {LETRAS}")

# === FUNCIÓN PARA EXTRAER FEATURES CON MEDIAPIPE ===
def extract_features_mediapipe(image):
    """Extrae 21 puntos de la mano (x, y, z) con MediaPipe"""
    try:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return None  # No se detectó mano

        landmarks = results.multi_hand_landmarks[0]
        features = []
        for lm in landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])  # 21 puntos * 3 coords = 63 features

        return np.array(features)
    except Exception as e:
        print(f"❌ Error en extracción con MediaPipe: {e}")
        return None

# === EXPLORAR DATASET ===
def explorar_dataset():
    print(f"\n🔍 Explorando: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        print("❌ La carpeta del dataset no existe")
        return
    carpetas_letras = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
    for carpeta in sorted(carpetas_letras):
        total_imgs = sum([len(glob(os.path.join(DATASET_PATH, carpeta, ext))) for ext in ['*.jpg','*.jpeg','*.png']])
        print(f"   📂 {carpeta}: {total_imgs} imágenes")

# === PROCESAR DATASET ===
def procesar_dataset_corregido():
    print(f"\n📸 Procesando dataset en: {DATASET_PATH}")
    for letra in LETRAS:
        carpeta_path = os.path.join(DATASET_PATH, letra)
        if not os.path.exists(carpeta_path):
            print(f"   ❌ No existe carpeta para {letra}")
            continue

        imagenes = []
        for ext in ['*.jpg','*.jpeg','*.png']:
            imagenes.extend(glob(os.path.join(carpeta_path, ext)))

        if not imagenes:
            print(f"   ❌ No hay imágenes en {carpeta_path}")
            continue

        print(f"\n🔤 Procesando {letra} con {len(imagenes)} imágenes")
        all_rows = []
        CSV_SALIDA = os.path.join(DATA_DIR, f'dataset_{letra}.csv')

        for i, img_path in enumerate(imagenes):
            image = cv2.imread(img_path)
            if image is None:
                print(f"   ⚠️ No se pudo leer {img_path}")
                continue

            features = extract_features_mediapipe(image)
            if features is None:
                continue

            row = np.append(features, letra.upper())
            all_rows.append(row)

            if (i+1) % 10 == 0:
                print(f"   ✅ Procesadas {i+1}/{len(imagenes)}")

        if all_rows:
            feat_len = len(all_rows[0]) - 1
            cols = [f'feature_{i}' for i in range(feat_len)] + ['label']
            df = pd.DataFrame(all_rows, columns=cols)
            df.to_csv(CSV_SALIDA, index=False)
            print(f"   💾 Guardado {CSV_SALIDA} con {len(df)} muestras")
        else:
            print(f"   ❌ Ninguna muestra válida para {letra}")

# === ENTRENAR MODELO ===
def entrenar_modelo():
    archivos = glob(os.path.join(DATA_DIR, 'dataset_*.csv'))
    if not archivos:
        print("❌ No hay CSV procesados.")
        return

    dfs = [pd.read_csv(archivo) for archivo in archivos]
    df = pd.concat(dfs, ignore_index=True)
    X = df.drop('label', axis=1).values
    y = df['label'].values

    print(f"\n📈 Datos totales: {len(X)} muestras")
    print(f"🔤 Clases: {sorted(np.unique(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("🎯 Entrenando Random Forest...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n📊 Precisión: {acc:.2%}")
    print(classification_report(y_test, y_pred))

    modelo_path = os.path.join(MODEL_DIR, 'modelo_abecedario_mediapipe.joblib')
    joblib.dump({'modelo': clf, 'scaler': scaler, 'labels': sorted(np.unique(y))}, modelo_path)
    print(f"💾 Modelo guardado en: {modelo_path}")

# === MENÚ ===
def main():
    while True:
        print("\n" + "="*60)
        print("🎯 PROCESADOR DE ABECEDARIO LSC (MediaPipe)")
        print("="*60)
        print("1. 🔍 Explorar dataset")
        print("2. 📸 Procesar imágenes")
        print("3. 🎯 Entrenar modelo")
        print("4. 🚪 Salir")
        opcion = input("\nSelecciona opción (1-4): ")

        if opcion == '1': explorar_dataset()
        elif opcion == '2': procesar_dataset_corregido()
        elif opcion == '3': entrenar_modelo()
        elif opcion == '4': break
        else: print("❌ Opción inválida")

if __name__ == "__main__":
    main()
