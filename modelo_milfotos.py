import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# ==============================
# CONFIGURACIÓN
# ==============================
DATASET_DIR = r"C:\Users\raven\Proyecto_de_profundizacion_traduccion_de_senas\Traductor_de_senas\datasets_pocas_fotos"
OUTPUT_DIR = r"C:\Users\raven\Proyecto_de_profundizacion_traduccion_de_senas\modelo_mil_fotos"
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
VAL_SPLIT = 0.3  # 30% para validación

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Entrenando en: {device}")

# ==============================
# TRANSFORMACIONES DE DATOS
# ==============================
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# CARGA DEL DATASET
# ==============================
dataset = datasets.ImageFolder(DATASET_DIR, transform=data_transforms)

val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"📂 Total imágenes: {len(dataset)}")
print(f"   🏋️‍♂️ Train: {len(train_dataset)}")
print(f"   ✅ Val: {len(val_dataset)}")
print(f"   📚 Clases: {len(dataset.classes)} -> {dataset.classes}")

# ==============================
# DEFINICIÓN DEL MODELO
# ==============================
model = models.resnet18(weights="IMAGENET1K_V1")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==============================
# ENTRENAMIENTO
# ==============================
best_acc = 0.0
os.makedirs(OUTPUT_DIR, exist_ok=True)

for epoch in range(EPOCHS):
    print(f"\n📌 Epoch {epoch+1}/{EPOCHS}")

    # ---- Entrenamiento ----
    model.train()
    running_loss, running_corrects = 0.0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / train_size
    epoch_acc = running_corrects.double() / train_size
    print(f"   🏋️‍♂️ Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # ---- Validación ----
    model.eval()
    val_loss, val_corrects = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

    val_loss /= val_size
    val_acc = val_corrects.double() / val_size
    print(f"   ✅ Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # Guardar mejor modelo
    if val_acc > best_acc:
        best_acc = val_acc
        model_path = os.path.join(OUTPUT_DIR, "resnet18_milfotos.pth")
        torch.save(model.state_dict(), model_path)
        print(f"   💾 Modelo guardado: {model_path}")

print(f"\n🎯 Entrenamiento terminado. Mejor accuracy en validación: {best_acc:.4f}")
