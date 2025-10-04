import os  # Operaciones del sistema operativo (no se usa en este script pero es común tenerlo)
import torch  # Librería principal para deep learning
import torch.nn as nn  # Submódulo para redes neuronales
import torch.optim as optim  # Submódulo para optimizadores
from torchvision import datasets, transforms, models  # Utilidades para imágenes y modelos preentrenados
from torch.utils.data import DataLoader, random_split  # Utilidades para cargar y dividir datasets

# ================================
# CONFIGURACIÓN DE PARÁMETROS
# ================================
DATASET_DIR = r"C:\Users\raven\Proyecto_de_profundizacion_traduccion_de_senas\Traductor_de_senas\datasets"  # Ruta del dataset
BATCH_SIZE = 32  # Tamaño de lote para entrenamiento y validación
EPOCHS = 15  # Número de épocas de entrenamiento
LR = 0.001  # Tasa de aprendizaje para el optimizador
VAL_SPLIT = 0.2  # Porcentaje de datos para validación (20%)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Usa GPU si está disponible, si no CPU
print(f"⚡ Entrenando en: {device}")  # Muestra el dispositivo usado

# ================================
# TRANSFORMACIONES DE DATOS
# ================================
data_transforms = transforms.Compose([  # Composición de transformaciones para las imágenes
    transforms.Resize((224, 224)),  # Redimensiona la imagen a 224x224 píxeles
    transforms.RandomHorizontalFlip(),  # Aplica flip horizontal aleatorio
    transforms.RandomRotation(10),  # Aplica rotación aleatoria de hasta 10 grados
    transforms.ToTensor(),  # Convierte la imagen a tensor
    transforms.Normalize([0.485, 0.456, 0.406],  # Normaliza usando la media de ImageNet
                         [0.229, 0.224, 0.225])  # y la desviación estándar de ImageNet
])

# ================================
# CARGA DEL DATASET
# ================================
dataset = datasets.ImageFolder(DATASET_DIR, transform=data_transforms)  # Carga las imágenes y aplica las transformaciones

# Divide el dataset en entrenamiento y validación
val_size = int(len(dataset) * VAL_SPLIT)  # Calcula el tamaño del conjunto de validación
train_size = len(dataset) - val_size  # Calcula el tamaño del conjunto de entrenamiento
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])  # Divide el dataset

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # Crea el DataLoader para entrenamiento
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)  # Crea el DataLoader para validación

print(f"📂 Total imágenes: {len(dataset)}")  # Muestra el total de imágenes
print(f"   🏋️‍♂️ Train: {len(train_dataset)}")  # Muestra el total de imágenes de entrenamiento
print(f"   ✅ Val: {len(val_dataset)}")  # Muestra el total de imágenes de validación
print(f"   📚 Clases: {len(dataset.classes)} -> {dataset.classes}")  # Muestra las clases detectadas

# ================================
# DEFINICIÓN DEL MODELO
# ================================
model = models.resnet18(weights="IMAGENET1K_V1")  # Carga el modelo ResNet18 preentrenado en ImageNet
num_ftrs = model.fc.in_features  # Obtiene el número de características de entrada de la capa final
model.fc = nn.Linear(num_ftrs, len(dataset.classes))  # Reemplaza la capa final para que tenga tantas salidas como clases
model = model.to(device)  # Mueve el modelo al dispositivo seleccionado (GPU o CPU)

criterion = nn.CrossEntropyLoss()  # Define la función de pérdida (entropía cruzada)
optimizer = optim.Adam(model.parameters(), lr=LR)  # Define el optimizador Adam

# Inicializa la mejor precisión en validación
best_acc = 0.0

# ================================
# ENTRENAMIENTO Y VALIDACIÓN
# ================================
for epoch in range(EPOCHS):  # Bucle principal de entrenamiento por épocas
    print(f"\n📌 Epoch {epoch+1}/{EPOCHS}")  # Muestra la época actual

    # ---- Entrenamiento ----
    model.train()  # Pone el modelo en modo entrenamiento
    running_loss, running_corrects = 0.0, 0  # Inicializa acumuladores de pérdida y aciertos
    for inputs, labels in train_loader:  # Itera sobre los lotes de entrenamiento
        inputs, labels = inputs.to(device), labels.to(device)  # Mueve los datos al dispositivo

        optimizer.zero_grad()  # Reinicia los gradientes
        outputs = model(inputs)  # Calcula la salida del modelo
        loss = criterion(outputs, labels)  # Calcula la pérdida
        loss.backward()  # Calcula los gradientes
        optimizer.step()  # Actualiza los pesos

        _, preds = torch.max(outputs, 1)  # Obtiene la clase predicha
        running_loss += loss.item() * inputs.size(0)  # Acumula la pérdida
        running_corrects += torch.sum(preds == labels.data)  # Acumula los aciertos

    epoch_loss = running_loss / train_size  # Calcula la pérdida promedio de la época
    epoch_acc = running_corrects.double() / train_size  # Calcula la precisión promedio de la época
    print(f"   🏋️‍♂️ Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")  # Muestra métricas de entrenamiento

    # ---- Validación ----
    model.eval()  # Pone el modelo en modo evaluación
    val_loss, val_corrects = 0.0, 0  # Inicializa acumuladores de pérdida y aciertos para validación
    with torch.no_grad():  # Desactiva el cálculo de gradientes (más eficiente)
        for inputs, labels in val_loader:  # Itera sobre los lotes de validación
            inputs, labels = inputs.to(device), labels.to(device)  # Mueve los datos al dispositivo
            outputs = model(inputs)  # Calcula la salida del modelo
            loss = criterion(outputs, labels)  # Calcula la pérdida

            _, preds = torch.max(outputs, 1)  # Obtiene la clase predicha
            val_loss += loss.item() * inputs.size(0)  # Acumula la pérdida
            val_corrects += torch.sum(preds == labels.data)  # Acumula los aciertos

    val_loss /= val_size  # Calcula la pérdida promedio de validación
    val_acc = val_corrects.double() / val_size  # Calcula la precisión promedio de validación
    print(f"   ✅ Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")  # Muestra métricas de validación

    # Guardar mejor modelo
    if val_acc > best_acc:  # Si la precisión de validación es la mejor hasta ahora
        best_acc = val_acc  # Actualiza la mejor precisión
        torch.save(model.state_dict(), "mejor_modelo_resnet18.pth")  # Guarda los pesos del modelo
        print("   💾 Modelo guardado (mejor hasta ahora)")  # Informa que se guardó el modelo

print(f"\n🎯 Entrenamiento terminado. Mejor accuracy en validación: {best_acc:.4f}")  # Muestra la mejor precisión final
