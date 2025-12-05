# Traductor de Lenguaje de Señas con IA

Reconocimiento en tiempo real del lenguaje de señas colombiano usando Deep Learning.

Desarrollado como proyecto de profundización universitario.

## Sobre el Proyecto

El Traductor de Lenguaje de Señas es una aplicación web que utiliza visión por computadora y aprendizaje profundo para reconocer en tiempo real las letras del alfabeto en lenguaje de señas colombiano (LSC).

Este proyecto busca facilitar la comunicación para la comunidad sorda mediante tecnología accesible y de código abierto.

## Caracteristicas

- **Reconocimiento en Tiempo Real:** Detecta letras del alfabeto instantáneamente usando tu webcam.
- **IA Avanzada:** Modelo ResNet18 fine-tuned con alta precisión.
- **Sistema de Estabilidad:** Confirma letras después de 2.5 segundos para mayor precisión.
- **Formación de Palabras:** Acumula letras y forma palabras automáticamente.
- **Autocompletado Inteligente:** Sugiere palabras basadas en un diccionario.
- **Corrección Contextual:** Corrige predicciones usando contexto lingüístico.
- **Interfaz Moderna:** Diseño intuitivo y visualmente agradable.
- **Text-to-Speech:** Síntesis de voz para reproducir las frases formadas.

## Requisitos Previos

- Python 3.8 o superior
- Webcam conectada
- Sistema Operativo: Windows, macOS o Linux

## Instalacion y Ejecucion

Sigue estos pasos para configurar y ejecutar el proyecto en tu computadora.

### 1. Clonar el Repositorio

```bash
git clone https://github.com/ravenstorm23/Traductor_de_se-as.git
cd Traductor_de_se-as
```

### 2. Configurar el Entorno Virtual

Es recomendable usar un entorno virtual para manejar las dependencias y evitar conflictos.

**En Windows:**

```bash
# Crear el entorno virtual
python -m venv .venv

# Activar el entorno virtual
.\.venv\Scripts\Activate
```

**En macOS / Linux:**

```bash
# Crear el entorno virtual
python3 -m venv .venv

# Activar el entorno virtual
source .venv/bin/activate
```

### 3. Instalar Dependencias

Una vez activado el entorno virtual (verás el nombre del entorno entre paréntesis en tu terminal), instala las librerías necesarias:

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la Aplicacion

Con el entorno virtual activo, ejecuta el archivo principal:

```bash
python app.py
```

La aplicación iniciará el servidor. Verás un mensaje en la terminal indicando que está corriendo (usualmente en `http://localhost:5000`).

### 5. Usar la Aplicacion

1. Abre tu navegador web (Google Chrome, Mozilla Firefox, Microsoft Edge, etc.).
2. Ingresa a la dirección: `http://localhost:5000`
3. Permite el acceso a la cámara cuando el navegador lo solicite.

## Uso del Sistema

- **Mostrar señas:** Coloca tu mano frente a la cámara.
- **Confirmar letra:** Mantén la seña estable por 2.5 segundos.
- **Finalizar palabra:** Presiona el botón "Finalizar Palabra" o la tecla "O".
- **Limpiar:** Presiona el botón "Limpiar Todo" o la tecla "C".
- **Autocompletar:** Haz clic en las palabras sugeridas que aparecen en pantalla.

## Configuracion

El archivo `app.py` contiene variables de configuración que puedes ajustar según tus necesidades:

- `CONF_THRESHOLD`: Umbral de confianza mínimo para detectar una seña.
- `STABLE_TIME`: Tiempo requerido para confirmar una letra.
- `NO_HAND_TIMEOUT`: Tiempo de espera para guardar palabra automáticamente si no se detecta mano.

## Documentacion Adicional

En la carpeta `docs/` encontrarás documentación técnica detallada:

- `ARCHITECTURE.md`: Detalles sobre la arquitectura del sistema y flujo de datos.
- `TRAINING.md`: Información sobre el dataset y el entrenamiento del modelo.
- `CONTRIBUTING.md`: Guía de contribución y créditos.

## Equipo

Este proyecto fue desarrollado por:

- Raven
- Mateo Rivera Maya
- Diego Fernando Fuentes

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.
