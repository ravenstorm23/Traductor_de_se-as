import os  # Operaciones del sistema operativo
import torch  # Librería principal para deep learning
import torch.nn as nn  # Submódulo para redes neuronales
from torch.utils.data import Dataset, DataLoader, random_split  # Utilidades para datasets y splits
import torchvision.transforms as transforms  # Transformaciones para imágenes
import torchvision.models as models  # Modelos preentrenados
import cv2  # OpenCV para manejo de video
import numpy as np  # Operaciones numéricas (no se usa explícitamente aquí)
from glob import glob  # Para buscar archivos por patrón
from tqdm import tqdm  # Barra de progreso

# ========================
# CONFIGURACIÓN DE PARÁMETROS
# ========================
DATASET_DIR = r"C:\Users\raven\Proyecto_de_profundizacion_traduccion_de_senas\Traductor_de_senas\datasets_dinamicos"  # Ruta del dataset de videos
MODEL_DIR = r"C:\Users\raven\Proyecto_de_profundizacion_traduccion_de_senas\Traductor_de_senas\modelos_abecedario"  # Ruta para guardar modelos
os.makedirs(MODEL_DIR, exist_ok=True)  # Crea la carpeta de modelos si no existe

LETRAS = ["G", "J", "S", "Z", "Ñ"]  # Clases a reconocer
BATCH_SIZE = 2  # Tamaño de lote
EPOCHS = 10  # Número de épocas de entrenamiento
LR = 1e-4  # Tasa de aprendizaje
SEQ_LEN = 30  # Número de frames por video a usar
IMG_SIZE = 224  # Tamaño de imagen para el modelo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Usa GPU si está disponible
VAL_SPLIT = 0.2  # Porcentaje de datos para validación
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")  # Extensiones de video válidas

# ========================
# DATASET PERSONALIZADO PARA VIDEOS
# ========================
class VideoDataset(Dataset):
    def __init__(self, dataset_dir, letras, seq_len=SEQ_LEN, transform=None):
        self.samples = []  # Lista de rutas de videos
        self.labels = []  # Lista de etiquetas (índices de letras)
        self.seq_len = seq_len  # Frames por video
        self.transform = transform  # Transformaciones a aplicar

        for idx, letra in enumerate(letras):  # Itera sobre cada clase
            folder = os.path.join(dataset_dir, letra)  # Carpeta de la clase
            if not os.path.isdir(folder):
                print(f"⚠️ Carpeta no encontrada: {folder}")
                continue
            videos = [f for f in glob(os.path.join(folder, "*")) if f.lower().endswith(VIDEO_EXTS)]  # Busca videos válidos
            if len(videos) == 0:
                print(f"⚠️ No se encontraron videos en: {folder}")
            for vid in videos:
                self.samples.append(vid)  # Agrega ruta de video
                self.labels.append(idx)  # Agrega índice de clase

        if len(self.samples) == 0:
            raise ValueError("❌ No se encontraron videos en ninguna subcarpeta del dataset")

    def __len__(self):
        return len(self.samples)  # Número total de videos

    def __getitem__(self, idx):
        path = self.samples[idx]  # Ruta del video
        label = self.labels[idx]  # Etiqueta de la clase

        cap = cv2.VideoCapture(path)  # Abre el video
        frames = []  # Lista para almacenar frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Número total de frames en el video
        step = max(1, total_frames // self.seq_len)  # Paso para muestrear frames uniformemente

        for i in range(self.seq_len):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i*step)  # Posiciona el video en el frame deseado
            ret, frame = cap.read()  # Lee el frame
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convierte de BGR a RGB
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))  # Redimensiona el frame
            if self.transform:
                frame = self.transform(frame)  # Aplica transformaciones
            frames.append(frame)  # Agrega el frame a la lista

        cap.release()  # Libera el video

        while len(frames) < self.seq_len:  # Si faltan frames, repite el último
            frames.append(frames[-1].clone())

        frames = torch.stack(frames)  # Convierte la lista a tensor (seq_len, 3, H, W)
        return frames, label  # Devuelve los frames y la etiqueta

# ========================
# TRANSFORMACIONES DE IMAGEN
# ========================
transform = transforms.Compose([
    transforms.ToTensor(),  # Convierte la imagen a tensor
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])  # Normaliza como en ImageNet
])

# ========================
# DATA LOADERS Y SPLIT
# ========================
dataset = VideoDataset(DATASET_DIR, LETRAS, transform=transform)  # Crea el dataset
print(f"✅ Total videos encontrados: {len(dataset)}")
val_size = int(len(dataset) * VAL_SPLIT)  # Tamaño del set de validación
train_size = len(dataset) - val_size  # Tamaño del set de entrenamiento
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])  # Divide el dataset
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # DataLoader de entrenamiento
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)  # DataLoader de validación

# ========================
# MODELO: ResNet + LSTM
# ========================
class ResNetLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1, num_classes=len(LETRAS)):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Carga ResNet18 preentrenada
        modules = list(resnet.children())[:-1]  # Quita la capa fully connected
        self.resnet = nn.Sequential(*modules)  # Solo las capas convolucionales
        self.resnet.eval()  # Pone ResNet en modo evaluación
        for param in self.resnet.parameters():
            param.requires_grad = False  # Congela los pesos de ResNet

        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)  # LSTM para secuencias
        self.fc = nn.Linear(hidden_size, num_classes)  # Capa final para clasificación

    def forward(self, x):
        B, seq_len, C, H, W = x.size()  # Tamaño del batch y secuencia
        x = x.view(B*seq_len, C, H, W)  # Junta batch y secuencia para pasar por ResNet
        with torch.no_grad():
            feats = self.resnet(x)  # Extrae características con ResNet
        feats = feats.view(B, seq_len, -1)  # Vuelve a separar batch y secuencia
        out, _ = self.lstm(feats)  # Pasa por LSTM
        out = out[:, -1, :]  # Toma la salida del último frame
        out = self.fc(out)  # Pasa por la capa final
        return out

model = ResNetLSTM().to(DEVICE)  # Instancia el modelo y lo mueve al dispositivo
criterion = nn.CrossEntropyLoss()  # Función de pérdida
optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # Optimizador Adam

# ========================
# ENTRENAMIENTO Y VALIDACIÓN
# ========================
best_acc = 0.0  # Mejor accuracy de validación
best_model_path = os.path.join(MODEL_DIR, "restnet_videos_lstm_best.pth")  # Ruta para guardar el mejor modelo

for epoch in range(EPOCHS):
    # --- ENTRENAMIENTO ---
    model.train()  # Modo entrenamiento
    running_loss = 0.0
    correct = 0
    total = 0
    for videos, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{EPOCHS}"):
        videos = videos.to(DEVICE)  # Mueve videos al dispositivo
        labels = labels.to(DEVICE)  # Mueve etiquetas al dispositivo

        optimizer.zero_grad()  # Reinicia gradientes
        outputs = model(videos)  # Forward pass
        loss = criterion(outputs, labels)  # Calcula la pérdida
        loss.backward()  # Backpropagation
        optimizer.step()  # Actualiza los pesos

        running_loss += loss.item() * videos.size(0)  # Acumula la pérdida
        _, predicted = torch.max(outputs, 1)  # Predicción de la clase
        total += labels.size(0)  # Total de muestras
        correct += (predicted == labels).sum().item()  # Suma aciertos

    train_loss = running_loss / total  # Pérdida promedio de entrenamiento
    train_acc = correct / total * 100  # Precisión de entrenamiento

    # --- VALIDACIÓN ---
    model.eval()  # Modo evaluación
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():  # No calcula gradientes
        for videos, labels in val_loader:
            videos = videos.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * videos.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= val_total  # Pérdida promedio de validación
    val_acc = val_correct / val_total * 100  # Precisión de validación

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    # Guarda el mejor modelo según validación
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Mejor modelo guardado con val acc: {best_acc:.2f}%")

print(f"✅ Entrenamiento terminado. Mejor modelo: {best_model_path} con val acc: {best_acc:.2f}%")
