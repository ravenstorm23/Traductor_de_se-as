# Dataset y Entrenamiento del Modelo

##  Dataset

### Composición Original

El dataset utilizado para entrenar este modelo fue creado específicamente para este proyecto mediante:

- **Recopilación manual de fotografías** de las señas del alfabeto
- Uso de **múltiples personas** para capturar variabilidad
- Diferentes **condiciones de iluminación** y fondos
- Variaciones en **ángulos** y **distancias** de la cámara

> [!NOTE]
> El dataset original no está incluido en este repositorio debido a su gran tamaño (varios GB). Se documentan aquí las técnicas y proceso utilizado.

### Letras Soportadas

El modelo reconoce **22 letras** del alfabeto:

```
A, B, C, D, E, F, H, I, K, L, M, N, O, P, Q, R, T, U, V, W, X, Y
```

**Letras no incluidas:** G, J, Ñ, S, Z

> [!IMPORTANT]
> Estas letras fueron excluidas debido a que requieren **movimiento** (señas dinámicas) en el lenguaje de señas colombiano, mientras que este modelo está entrenado solo para **señas estáticas**.

##  Data Augmentation

Para mejorar la robustez y generalización del modelo, se aplicaron las siguientes técnicas de **data augmentation**:

### Transformaciones Geométricas

1. **Rotación Aleatoria**
   - Rango: ±15 grados
   - Propósito: Invarianza a inclinación de la mano

2. **Traslación Horizontal y Vertical**
   - Rango: ±10% del tamaño de la imagen
   - Propósito: Robustez a diferentes posiciones en el marco

3. **Escala (Zoom)**
   - Rango: 0.9x a 1.1x
   - Propósito: Invarianza a distancia de la cámara

4. **Volteo Horizontal (Flip)**
   - Probabilidad: 50%
   - Propósito: Reconocer mano derecha e izquierda

### Transformaciones de Color e Iluminación

5. **Ajuste de Brillo**
   - Rango: ±20%
   - Propósito: Adaptación a diferentes iluminaciones

6. **Ajuste de Contraste**
   - Rango: ±15%
   - Propósito: Mejorar definición en condiciones adversas

7. **Ajuste de Saturación**
   - Rango: ±10%
   - Propósito: Invarianza a tonos de piel

### Transformaciones de Ruido

8. **Ruido Gaussiano**
   - Desviación estándar: 0.01
   - Propósito: Robustez a compresión de video y artefactos

### Resultado del Augmentation

Cada imagen original generó aproximadamente **8-10 variaciones**, multiplicando el tamaño efectivo del dataset por esa cantidad.

##  Arquitectura del Modelo

### ResNet18

Se utilizó **ResNet18** pre-entrenado en ImageNet como base:

```python
model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 22)  # 22 clases (letras)
```

**Justificación de ResNet18:**
-  Balance entre precisión y velocidad
-  Pre-entrenado en ImageNet (transfer learning)
-  Arquitectura probada para clasificación de imágenes
-  Suficientemente ligero para inferencia en tiempo real

### Modificaciones

1. **Capa final:** Reemplazada para 22 clases (letras)
2. **Fine-tuning:** Todas las capas entrenadas (no solo la última)
3. **Normalización:** Usando estadísticas de ImageNet

##  Proceso de Entrenamiento

### Hiperparámetros

| Parámetro          | Valor            | Justificación                   |
| ------------------ | ---------------- | ------------------------------- |
| **Optimizador**    | Adam             | Convergencia rápida y estable   |
| **Learning Rate**  | 0.001            | Valor estándar para fine-tuning |
| **Batch Size**     | 32               | Balance memoria/velocidad       |
| **Epochs**         | 50-100           | Hasta convergencia              |
| **Loss Function**  | CrossEntropyLoss | Clasificación multiclase        |
| **Regularización** | Dropout (0.5)    | Prevenir overfitting            |

### División del Dataset

```
Dataset Total
├── Entrenamiento (70%): ~X,XXX imágenes
├── Validación (15%): ~XXX imágenes  
└── Prueba (15%): ~XXX imágenes
```

> [!NOTE]
> Las cantidades exactas dependen del número de fotos recopiladas originalmente.

### Early Stopping

- Monitoreo de pérdida en validación
- Paciencia: 10 epochs sin mejora
- Guardado del mejor modelo según precisión de validación

## 📈 Métricas de Evaluación

### Métricas Principales

1. **Accuracy** (Precisión global)
   - Porcentaje de predicciones correctas
   - Métrica principal para evaluar el modelo

2. **Precision, Recall, F1-Score por clase**
   - Identificar letras con baja precisión
   - Balancear falsos positivos y negativos

3. **Matriz de Confusión**
   - Identificar pares de letras confundidas frecuentemente
   - Ejemplo: B ↔ V, K ↔ R

### Resultados Esperados

En condiciones óptimas (buena iluminación, fondo simple):
- **Accuracy general:** ~85-95%
- **Confianza promedio:** >80%
- **Latencia de inferencia:** <50ms (CPU)

## 🔧 Preprocesamiento en Producción

Durante la inferencia en tiempo real, cada frame pasa por:

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),    # Tamaño de entrada ResNet
    transforms.ToTensor(),            # Convertir a tensor
    transforms.Normalize(             # Normalización ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

##  Técnicas Adicionales Implementadas

### 1. Center Cropping
- Enfoque en región central de la mano
- Reduce ruido del fondo

### 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Mejora contraste local
- Útil en condiciones de iluminación variable

### 3. ROI Extraction con MediaPipe
- Detección precisa de la mano
- Eliminación automática del fondo
- Margen adicional (40px) para capturar toda la seña

##  Script de Entrenamiento

El script `entrenar_fotos_RestNet18.py` (incluido en desarrollo, no en repo):

```python
# Estructura básica del entrenamiento
def train_epoch(model, dataloader, criterion, optimizer):
    for images, labels in dataloader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
def validate(model, dataloader):
    # Evaluar en validation set
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            # Calcular métricas
```

##  Mejoras Futuras

### Modelo
- [ ] Probar arquitecturas más modernas (EfficientNet, Vision Transformer)
- [ ] Implementar ensemble de modelos
- [ ] Cuantización del modelo para móviles
- [ ] Destilación de conocimiento

### Dataset
- [ ] Expandir a señas dinámicas (LSTM/Transformer)
- [ ] Incluir más variabilidad (edades, tonos de piel)
- [ ] Etiquetas con niveles de confianza
- [ ] Aumento con GAN (Generative Adversarial Networks)

### Técnicas
- [ ] Attention mechanisms
- [ ] Few-shot learning para nuevas señas
- [ ] Self-supervised learning

##  Referencias

### Papers Relevantes
- **ResNet:** "Deep Residual Learning for Image Recognition" (He et al., 2015)
- **Transfer Learning:** "A Survey on Transfer Learning" (Pan & Yang, 2010)
- **Data Augmentation:** "The Effectiveness of Data Augmentation in Image Classification" (Perez & Wang, 2017)

### Recursos Utilizados
- PyTorch Documentation
- MediaPipe Hands
- ImageNet Dataset (pre-training)

---

##  Lecciones Aprendidas

1. **Transfer Learning es crucial:** El pre-entrenamiento en ImageNet aceleró significativamente la convergencia

2. **Data Augmentation es esencial:** Multiplicó el dataset efectivo y mejoró la generalización

3. **ROI Extraction reduce ruido:** MediaPipe Hands permitió eliminar el fondo y enfocarse en la mano

4. **Estabilidad temporal es necesaria:** El sistema de confirmación de 2.5s reduce falsos positivos

5. **Corrección contextual ayuda:** El diccionario mejora la precisión al usar conocimiento lingüístico

---

**Nota:** Para replicar el entrenamiento, necesitarás crear tu propio dataset de señas o contactar a los autores para más información sobre el proceso de recopilación.
