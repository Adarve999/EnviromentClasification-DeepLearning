# Clasificador de Entornos

Creación de una **IA** capaz de recibir una imagen de entrada y clasificarla según su entorno o contexto (por ejemplo, `Kitchen`, `Office`, etc.).  

Este proyecto **entrena** diferentes **Redes Neuronales Convolucionales (CNN)** a partir de **modelos preentrenados** (ResNet, ResNeXt, RegNet, etc.), y aprovecha **Weights & Biases (W&B)** para registrar las métricas de entrenamiento de cada modelo.

## Descripción General

1. **Clases de entornos**  
   Se entrenó la IA para predecir múltiples ambientes. Los datos se encuentran en la carpeta `data`, organizada en subdirectorios para entrenamiento y validación.

2. **App con Streamlit**  
   El proyecto incluye una **app con Streamlit** que, de forma amigable, permite subir una imagen y clasificarla usando cualquiera de los modelos disponibles, mostrando un top 5 de probabilidades y la probabilidad de acierto.

3. **Monitoreo con Weights & Biases**  
   Todos los entrenamientos y sus métricas se reportan a W&B, facilitando la comparación de modelos, arquitecturas, épocas y otros hiperparámetros.

---

## Requerimientos Previos

1. **Clonar el repositorio**  
   ```bash
   git clone https://github.com/Adarve999/machineLearning_II.git
   cd machineLearning_II
   ```

2. **Crear un entorno virtual & instalar librerías**  
   ```bash
   python -m venv <entorno>
   <entorno>/Scripts/activate       # o source <entorno>/bin/activate en Linux/Mac
   pip install -r requirements.txt
   ```

3. **(Opcional) Ajustar .gitignore**  
   Si necesitas ignorar archivos sensibles o pesados, agrégalos en `.gitignore`.

> **Importante**:  
> - Si deseas entrenar en GPU, asegúrate de instalar **CUDA** y la versión adecuada de **Torch**/`torchvision`.  
> - Algunos modelos superan 100MB; si no se encuentran en la carpeta `models/`, podría requerirse descargarlos aparte (por ejemplo, desde Drive).

---

## Estructura de Carpetas

```
DeepLearning-ImageClassification/
├─ data/
│   ├─ training/
│   │   ├─ class1/
│   │   ├─ class2/
│   └─ validation/
│       ├─ class1/
│       ├─ class2/
│
├─ models/
│   ├─ (aquí se guardan los archivos .pt o .pth con los pesos entrenados)
│
├─ src/
│   ├─ train_model.py          # Script principal de entrenamiento
│   ├─ app_DragAndDrop.py      # App Streamlit para clasificar imágenes
│   └─ utils/
│       ├─ cnn.py              # Clases y funciones para la CNN
│       └─ ...
│
├─ wandb/                      # Directorio creado por Weights & Biases (con logs)
├─ README.md
├─ requirements.txt
└─ .gitignore
```

---

## Entrenamiento de Modelos

Para entrenar un modelo, ajusta los **hiperparámetros** (épocas, LR, capas descongeladas) en el script o desde la consola. Luego:

```bash
python src/train_model.py
```

- Se cargan los pesos preentrenados (p. ej., ImageNet).
- Se entrena en `data/training`, validando en `data/validation`.
- Se reportan métricas a W&B y se guarda el mejor modelo en la carpeta `models/`.

**Opciones de entrenamiento**  
- **Learning Rate**: Se puede ingresar en consola.  
- **Capas descongeladas**: También se especifican al inicio, por defecto 0.  
- **Número de épocas**: Preguntado al usuario, con un valor por defecto si no ingresa nada.

---

## Predicción con Streamlit

Para lanzar la app:

```bash
streamlit run src/app_DragAndDrop.py
```

1. Selecciona el modelo en el panel lateral.  
2. Sube una imagen (JPG, PNG…).  
3. El script aplica las transformaciones necesarias y ejecuta la clasificación.  
4. Muestra la clase predicha.

---

## Resultados y Métricas

A continuación, la **tabla con algunos resultados** de modelos entrenados, con sus épocas, tasa de aprendizaje, capas descongeladas y exactitudes (Train y Validación). Estos porcentajes son orientativos y dependen del dataset y configuraciones finales:

| Modelo              | Épocas | Learning Rate | Unfrozen Layers | Accuracy (Train) | Accuracy (Validación) |
|---------------------|-------:|--------------:|----------------:|-----------------:|-----------------------:|
| resnet50            |     50 | 0,0001        |               0 |           80,00% |                 80,00% |
| resnet50            |     30 | 0,0001        |               5 |           86,00% |                 87,00% |
| resnext101_32x8d    |    100 | 0,0001        |               0 |           84,00% |                 92,00% |
| resnext101_32x8d    |    200 | 0,0001        |               0 |           88,00% |                 92,30% |
| resnext101_32x8d    |     30 | 0,0001        |              13 |           93,30% |                 93,60% |
| resnext101_64x4d    |     30 | 0,0001        |               9 |           93,40% |                 94,20% |
| regnet_y_32gf       |     10 | 0,0001        |               0 |           71,00% |                 73,00% |
| regnet_y_32gf       |     10 | 0,0001        |               5 |           86,70% |                 94,70% |
| regnet_y_32gf       |     30 | 0,0001        |               7 |           91,90% |                 94,40% |

Como se observa, **ResNeXt** y **RegNet** en configuraciones avanzadas suelen obtener mejor rendimiento, aunque depende del conjunto de datos y la cantidad de capas descongeladas.

---

## Uso con GPU

Si quieres entrenar con GPU, asegúrate de:

```bash
conda install cudatoolkit
# o pip install torch==<version>+cu117 -f https://download.pytorch.org/whl/cu117
```

Revisa la compatibilidad de CUDA con tu versión de PyTorch. De lo contrario, se ejecutará en **CPU** y tomará más tiempo.

---