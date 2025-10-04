# evalar_30pct_val.py
import os
import csv
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from collections import defaultdict

# ---------- CONFIG ----------
DATASET_DIR = r"C:\Users\raven\Proyecto_de_profundizacion_traduccion_de_senas\Traductor_de_senas\datasets"
model_path  = r"C:\Users\raven\Proyecto_de_profundizacion_traduccion_de_senas\Traductor_de_senas\modelos_abecedario\mejor_modelo_resnet18.pth"
BATCH_SIZE  = 32
VAL_RATIO   = 0.30   # 30% para validación
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
# ----------------------------

print(f"🔎 Evaluación en: {DEVICE}")
assert os.path.isdir(DATASET_DIR), f"Dataset no encontrado en: {DATASET_DIR}"
assert os.path.exists(model_path), f"Modelo no encontrado en: {model_path}"

# ---------- TRANSFORMACIONES (deterministas para validación) ----------
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------- DATASET y SPLIT ----------
full_dataset = datasets.ImageFolder(DATASET_DIR, transform=val_transform)
total_len = len(full_dataset)
val_len = int(total_len * VAL_RATIO)
train_len = total_len - val_len
generator = torch.Generator().manual_seed(SEED)
train_ds, val_ds = random_split(full_dataset, [train_len, val_len], generator=generator)

print(f"📂 Total imágenes: {total_len}")
print(f"   🏋️‍ Train (no usado aquí): {len(train_ds)}")
print(f"   ✅ Val (usado para test): {len(val_ds)}")
print(f"   📚 Clases detectadas: {len(full_dataset.classes)} -> {full_dataset.classes}")

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------- MODELO ----------
# Usamos la misma construcción que en tu script de entrenamiento
model = models.resnet18(weights="IMAGENET1K_V1")
num_classes = len(full_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

# Cargar pesos (state_dict guardado en entrenamiento)
try:
    state = torch.load(model_path, map_location=DEVICE)
except Exception as e:
    raise RuntimeError(f"Error cargando el modelo: {e}")
model.load_state_dict(state)
model.eval()

# ---------- EVALUACIÓN ----------
confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion[t.long(), p.long()] += 1

total = confusion.sum().item()
correct = confusion.diag().sum().item()
overall_acc = correct / total if total > 0 else 0.0

print(f"\n🎯 Resultados (sobre {val_len} imágenes):")
print(f"   ✅ Correctos: {correct}")
print(f"   📊 Accuracy global: {overall_acc:.4f}")

# Per-class accuracy y guardado CSV
report_rows = [["class", "correct", "total", "accuracy"]]
for i, cls_name in enumerate(full_dataset.classes):
    class_correct = confusion[i, i].item()
    class_total = confusion[i, :].sum().item()
    class_acc = (class_correct / class_total) if class_total > 0 else 0.0
    print(f"   - {cls_name}: {class_correct}/{class_total}  acc={class_acc:.4f}")
    report_rows.append([cls_name, class_correct, class_total, f"{class_acc:.4f}"])

csv_out = "validation_report_30pct.csv"
with open(csv_out, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(report_rows)

print(f"\n💾 Reporte guardado en: {csv_out}")
