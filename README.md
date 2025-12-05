<div align="center">

# Traductor de Lenguaje de Señas con IA

### Reconocimiento en tiempo real del lenguaje de señas colombiano usando Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Desarrollado como proyecto de profundización universitario*

[Características](#-características) • [Demo](#-demo) • [Instalación](#-instalación) • [Uso](#-uso) • [Documentación](#-documentación) • [Equipo](#-equipo)

</div>

---

##  Sobre el Proyecto

El **Traductor de Lenguaje de Señas** es una aplicación web que utiliza visión por computadora y aprendizaje profundo para reconocer en tiempo real las letras del alfabeto en lenguaje de señas colombiano (LSC). 

Este proyecto busca **facilitar la comunicación** para la comunidad sorda mediante tecnología accesible y de código abierto.

###  Objetivos

-  Reconocer 22 letras del alfabeto en LSC en tiempo real
-  Proporcionar una interfaz web intuitiva y accesible
-  Implementar corrección contextual con diccionario
-  Ofrecer autocompletado inteligente de palabras
-  Contribuir a la inclusión digital

---

##  Características

###  Funcionalidades Principales

- ** Reconocimiento en Tiempo Real:** Detecta letras del alfabeto instantáneamente usando tu webcam
- ** IA Avanzada:** Modelo ResNet18 fine-tuned con >85% de precisión
- ** Sistema de Estabilidad:** Confirma letras después de 2.5 segundos para mayor precisión
- ** Formación de Palabras:** Acumula letras y forma palabras automáticamente
- ** Autocompletado Inteligente:** Sugiere palabras basadas en un diccionario de 6,900+ términos
- ** Corrección Contextual:** Corrige predicciones usando contexto lingüístico
- ** Interfaz Premium:** Diseño moderno con gradientes, animaciones y efectos visuales
- ** Text-to-Speech:** Síntesis de voz para reproducir las frases formadas
- ** Atajos de Teclado:** `O` para finalizar palabra, `C` para limpiar

###  Interfaz de Usuario

- Video en vivo con overlay de información
- Indicador visual de estabilidad (barra de progreso)
- Sugerencias de autocompletado en tiempo real
- Indicadores de confianza por colores:
  - 🟢 Verde = Alta confianza (≥80%)
  - 🔴 Rojo = Baja confianza (<80%)

---

##  Demo

> [!NOTE]
> Aquí puedes agregar un GIF o screenshot de la aplicación en funcionamiento

```
[Espacio para screenshot o GIF animado]
```

**Letras soportadas:** A, B, C, D, E, F, H, I, K, L, M, N, O, P, Q, R, T, U, V, W, X, Y

*(Las letras G, J, Ñ, S, Z requieren movimiento y no están incluidas en esta versión)*

---

##  Instalación

### Requisitos Previos

- **Python 3.8 o superior**
- **Webcam** conectada
- **Sistema Operativo:** Windows, macOS o Linux

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/ravenstorm23/Traductor_de_se-as.git
cd Traductor_de_se-as
```

### Paso 2: Crear Entorno Virtual (Recomendado)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Verificar Modelo

Asegúrate de que el modelo entrenado esté en:
```
modelos_abecedario/mejor_modelo_resnet18.pth
```

> [!IMPORTANT]
> El modelo pre-entrenado debe descargarse por separado si no está incluido. Contacta a los autores para obtenerlo.

---

##  Uso

### Iniciar la Aplicación

```bash
python app.py
```

La aplicación se iniciará en: **http://localhost:5000**

### Cómo Usar

1. **Abre tu navegador** y navega a `http://localhost:5000`
2. **Permite el acceso a la cámara** cuando se solicite
3. **Muestra las señas** frente a la cámara
4. **Espera 2.5 segundos** con la misma seña para que se confirme la letra
5. **Las letras se acumularán** formando una palabra
6. **Finaliza la palabra:**
   - Presiona el botón **"Finalizar Palabra"**
   - Presiona la tecla `O`
   - Espera 3.5 segundos sin mostrar señas (finalización automática)
7. **Usa el autocompletado** haciendo clic en las sugerencias
8. **Reproduce la frase** con el botón **"🔊 Hablar Frase"**
9. **Limpia todo** con el botón **"Limpiar Todo"** o la tecla `C`

### Atajos de Teclado

| Tecla | Acción                        |
| ----- | ----------------------------- |
| `O`   | Finalizar palabra actual      |
| `C`   | Limpiar todo (reset completo) |

---

## ⚙️ Configuración Avanzada

Puedes modificar parámetros en `app.py`:

```python
CONF_THRESHOLD = 0.80      # Umbral de confianza (default: 80%)
STABLE_TIME = 2.5          # Segundos para confirmar letra
FRAMES_CONSISTENTES = 8    # Frames consecutivos requeridos
NO_HAND_TIMEOUT = 3.5      # Segundos sin mano para guardar palabra
CLEAR_TIMEOUT = 10.0       # Segundos de inactividad para limpiar
```

---

##  Arquitectura Técnica

### Stack Tecnológico

| Componente                 | Tecnología                        |
| -------------------------- | --------------------------------- |
| **Backend**                | Flask 3.x                         |
| **Deep Learning**          | PyTorch 2.x                       |
| **Modelo**                 | ResNet18 (pre-entrenado ImageNet) |
| **Detección de Manos**     | MediaPipe Hands                   |
| **Procesamiento de Video** | OpenCV 4.x                        |
| **Frontend**               | HTML5, CSS3, Vanilla JavaScript   |
| **Síntesis de Voz**        | pyttsx3                           |

### Flujo de Datos

```
Webcam → MediaPipe (Detección) → ROI Extraction → ResNet18 (Clasificación)
   ↓
Sistema de Estabilidad → Corrección Contextual → Autocompletado
   ↓
Estado de Aplicación → API REST → Frontend (UI)
```

Para más detalles, consulta [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

##  Documentación

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Arquitectura del sistema, flujo de datos, componentes
- **[TRAINING.md](docs/TRAINING.md)** - Dataset, data augmentation, proceso de entrenamiento
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Guía para contribuir y créditos del equipo

---

##  Dataset y Entrenamiento

### Dataset

- **Origen:** Recopilación manual de fotografías
- **Técnica:** Data augmentation extensiva (rotación, traslación, ajustes de color, ruido)
- **Clases:** 22 letras del alfabeto LSC
- **Tamaño:** Dataset original multiplicado ~8-10x con augmentation

> [!NOTE]
> El dataset original no está incluido por su gran tamaño. Ver [docs/TRAINING.md](docs/TRAINING.md) para detalles del proceso.

### Modelo

- **Arquitectura:** ResNet18 pre-entrenado en ImageNet
- **Fine-tuning:** Todas las capas entrenadas
- **Precisión:** ~85-95% en condiciones óptimas
- **Latencia:** <50ms en CPU, <10ms en GPU

---

##  Equipo

Este proyecto fue desarrollado por:

- **Raven** - [@ravenstorm23](https://github.com/ravenstorm23)
- **Mateo Rivera Maya**
- **Diego Fernando Fuentes**

Como parte de un **proyecto de profundización universitario** enfocado en inteligencia artificial y accesibilidad.

---

##  Contribuciones

¡Las contribuciones son bienvenidas! Por favor lee [CONTRIBUTING.md](CONTRIBUTING.md) para conocer el proceso.

### Áreas de Mejora

- [ ] Ampliar a señas dinámicas (con movimiento)
- [ ] Soporte para más idiomas de señas
- [ ] Optimización del modelo (cuantización, TensorRT)
- [ ] Aplicación móvil (iOS/Android)
- [ ] API REST pública
- [ ] Modo offline

---

##  Licencia

Este proyecto está licenciado bajo la **Licencia MIT** - ver [LICENSE](LICENSE) para detalles.

Esto significa que puedes usar, modificar y distribuir este código libremente, siempre que mantengas el aviso de copyright.

---

##  Reconocimientos

- **MediaPipe** por su excelente biblioteca de detección de manos
- **PyTorch** por el framework de deep learning
- **Comunidad sorda** por inspirar este proyecto
- **ImageNet** por el dataset de pre-entrenamiento

---

##  Contacto

Para preguntas, sugerencias o colaboraciones:

- Abre un [Issue](https://github.com/ravenstorm23/Traductor_de_se-as/issues) en GitHub
- Contacta a través de LinkedIn (ver perfil de autores)

---

<div align="center">

###  Si este proyecto te fue útil, considera darle una estrella en GitHub

**Desarrollado con  para la comunidad sorda**

[⬆ Volver arriba](#-traductor-de-lenguaje-de-señas-con-ia)

</div>
