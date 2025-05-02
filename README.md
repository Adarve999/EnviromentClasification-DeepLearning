# Clasificador de Entornos

Creación de una **IA** capaz de recibir una imagen de entrada y clasificarla según su entorno o contexto (por ejemplo, `Kitchen`, `Office`, etc.).  

Este proyecto **entrena** diferentes **Redes Neuronales Convolucionales (CNN)** a partir de **modelos preentrenados** (ResNet, ResNeXt, RegNet, etc.), y aprovecha **Weights & Biases (W&B)** para registrar las métricas de entrenamiento de cada modelo.

## Descripción General

1. **Clases de entornos**  
   Se entrenó la IA para predecir múltiples ambientes. Los datos se encuentran en la carpeta `data`, organizada en subdirectorios para entrenamiento y validación.

2. **App con Streamlit**  
   El proyecto incluye una **app con Streamlit** que, de forma amigable, permite subir una imagen y clasificarla usando cualquiera de los modelos disponibles.

3. **Monitoreo con Weights & Biases**  
   Todos los entrenamientos y sus métricas se reportan a W&B, facilitando la comparación de modelos, arquitecturas, épocas y otros hiperparámetros.

---

# Ejecución Local Desde Cero

## 1. Clona el repositorio

```bash
git clone https://github.com/Adarve999/EnviromentClasification-DeepLearning.git
cd EnviromentClasification-DeepLearning
```

## 2. Crea un entorno virtual e instala las dependencias

```bash
conda env create -f env_dev.yaml
conda env activate dl_clasification
```

## 3 Entrenamiento o reutilizar modelos guardados en Google Drive

### 3.a Entrena un modelo

```bash
python src/train_model.py
```

1. Carga pesos preentrenados.
2. Entrena en `data/training`, validando en `data/validation`.
3. Reporta métricas a W&B (si lo configuras) y guarda el mejor modelo en `models/`.

#### Opciones de entrenamiento

- **Learning Rate**: Se solicita en consola; por defecto 1e-4.  
- **Capas descongeladas**: También se ingresa en consola; por defecto 0.  
- **Número de épocas**: Si no se ingresa, por defecto 1.

### 3.b Coger un modelo preentrenado de la carpeta de Google Drive

Debido a que algunos de los modelos entrenados superan el límite de 100 MB impuesto por GitHub, se han subido a una carpeta de Google Drive para facilitar su descarga. Puedes acceder a todos los modelos en el siguiente enlace:

[Carpeta de Drive con los modelos](https://drive.google.com/drive/folders/1-9ZGXn6zHftPIvnO7t1JfCHSlHR5p2DT?usp=sharing)

Para utilizarlos:

1. Descarga los archivos `.pt` necesarios.
2. Colócalos en la carpeta `models/` de este repositorio.

## 4. Lanza la app de clasificación con Streamlit

Una vez entrenado (o descargado) un modelo:

```bash
streamlit run src/app_DragAndDrop.py
```

1. Elige el modelo en el panel lateral.  
2. Sube la imagen.  
3. Se realiza la clasificación y se muestra la clase predicha.

---

# Resultados y Métricas

La siguiente tabla resume algunos resultados de distintos modelos, con sus épocas, tasa de aprendizaje, capas descongeladas y exactitudes (Train y Validación). Estos valores pueden variar según el dataset y la configuración final:

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

A partir de las métricas presentadas en la tabla, se pueden extraer varias conclusiones:

1. **Eficacia de ResNeXt y RegNet:**  
   Los modelos **ResNeXt** (especialmente con configuraciones 32x8d y 64x4d) y **RegNet** (en configuraciones con más capas descongeladas) suelen obtener mejores resultados de exactitud, tanto en entrenamiento como en validación, superando en varios puntos porcentuales a ResNet50.

2. **Importancia de descongelar capas:**  
   Al comparar, por ejemplo, el modelo **resnet50** con 0 capas descongeladas (80%–80%) frente a **resnet50** con 5 capas descongeladas (86%–87%), se observa un aumento significativo en la exactitud. Esto indica que **ajustar capas internas** (unfreeze) puede permitir al modelo aprender mejor las características específicas del dataset.

3. **Mayor número de épocas no siempre significa mayor exactitud:**  
   En la familia **ResNeXt101_32x8d**, se ven resultados para 100 y 200 épocas con (casi) la misma validación (~92%), mientras que con apenas 30 épocas y 13 capas descongeladas sube a ~93,6%. Esto sugiere que la **calidad del ajuste** (qué capas se descongelan y con qué LR) puede ser más determinante que simplemente entrenar más tiempo.

4. **Aprendizaje efectivo cuando se combina buena arquitectura + capas descongeladas:**  
   Modelos como **resnext101_64x4d** con 9 capas descongeladas logran un 93,4% en training y 94,2% en validación, lo que demuestra la eficacia de descongelar parte de la red y elegir una arquitectura con alta cardinalidad.

5. **RegNet mejora significativamente con más capas descongeladas y más épocas:**  
   Pasar de 0 capas descongeladas (71%–73%) a 5 capas (86,7%–94,7%) o 7 capas (91,9%–94,4%) muestra un salto grande en rendimiento, confirmando que la estrategia de ajuste incide mucho en la exactitud final.

---

# Authors

- Rubén Adarve Pérez
- Marta Rodríguez Hebles
- Maria Valvanera Gil de Biedma
- Blanca Sayas Ladaga

Please use this bibtex if you want to cite this repository (master branch) in your publications:

```
@misc{EnviromentClasification-DeepLearning,
  author       = {Rubén Adarve Pérez, Marta Rodríguez Hebles, Maria Valvanera Gil de Biedma, Blanca Sayas Ladaga},
  title        = {Deep Learning App: Clasificador de Entornos},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{<https://github.com/Adarve999/EnviromentClasification-DeepLearning}}>,
}
```