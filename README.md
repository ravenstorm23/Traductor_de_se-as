#  Traductor de Señas IA - Interfaz Web

Una aplicación web moderna y elegante para la traducción de lenguaje de señas en tiempo real usando ResNet18.

##  Características

- **Reconocimiento en tiempo real**: Detecta letras del alfabeto en lenguaje de señas
- **Interfaz premium**: Diseño moderno con gradientes, animaciones y efectos visuales
- **Sistema de estabilidad**: Confirma letras después de 4 segundos de estabilidad
- **Formación de palabras**: Acumula letras y forma palabras completas
- **Controles intuitivos**: Botones para finalizar palabras y limpiar
- **Atajos de teclado**: `O` para finalizar palabra, `C` para limpiar

##  Requisitos Previos

- Python 3.8 o superior
- Webcam conectada
- Modelo entrenado ResNet18 (ubicado en `modelos_abecedario/mejor_modelo_resnet18.pth`)

##  Instalación

1. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

## 💻 Uso

1. **Iniciar el servidor**:
```bash
python app.py
```

2. **Abrir el navegador**:
   - Navega a: `http://localhost:5000`

3. **Usar la aplicación**:
   - El sistema automáticamente detectará las señas frente a la cámara
   - Las letras se confirman después de 4 segundos de estabilidad
   - Usa el botón **"Finalizar Palabra"** o presiona `O` para guardar la palabra actual
   - Usa el botón **"Limpiar Todo"** o presiona `C` para resetear todo

##  Características de la Interfaz

- **Video en vivo**: Muestra la cámara con overlay de información
- **Palabra actual**: Visualiza las letras detectadas en tiempo real
- **Indicador de estabilidad**: Muestra cuando una letra está siendo confirmada
- **Frase completa**: Acumula todas las palabras formadas
- **Indicadores visuales**: 
  - Verde = Alta confianza (≥85%)
  - Rojo = Baja confianza (<85%)

##  Configuración

Puedes modificar los siguientes parámetros en `app.py`:

- `CONF_THRESHOLD`: Umbral de confianza (default: 0.85)
- `BUFFER_SIZE`: Frames para suavizado (default: 7)
- `TIEMPO_ESTABILIDAD`: Segundos para confirmar letra (default: 4.0)
- `MODEL_PATH`: Ruta al modelo entrenado

## 📱 Tecnologías Utilizadas

- **Backend**: Flask, PyTorch, OpenCV
- **Frontend**: HTML5, CSS3, JavaScript
- **Modelo**: ResNet18 pre-entrenado
- **Diseño**: Inter font, Font Awesome icons

##  Letras Soportadas

A, B, C, D, E, F, H, I, K, L, M, N, O, P, Q, R, T, U, V, W, X, Y

---

**Desarrollado  para la comunidad sorda**
