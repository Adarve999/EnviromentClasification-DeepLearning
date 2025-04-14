# DeepLearning-ImageClassification

Este repositorio contiene ejemplos y scripts para **entrenar** y **predecir** con diferentes modelos de clasificación de imágenes usando **PyTorch**. Los modelos se han seleccionado tanto de la familia ResNet/ResNeXt como de la familia RegNet, aprovechando sus pesos preentrenados (en ImageNet) para llevar a cabo un _fine tuning_ a un conjunto de datos personalizado.

## Modelos Usados

1. **ResNet50**  
   - Modelo clásico ampliamente usado en visión por computadora.  
   - 50 capas con bloques _bottleneck_ y conexiones residuales.

2. **ResNeXt101_32x8d**  
   - Variante de ResNet que introduce la “cardinalidad” (32 grupos, cada uno de 8 canales).  
   - Aumenta la capacidad de la red sin crecer excesivamente en parámetros.

3. **ResNeXt101_64x4d**  
   - Otra configuración de cardinalidad (64 grupos, 4 canales por grupo).  
   - Similar objetivo: más rutas en paralelo para mayor expresividad.

4. **RegNetY_32GF**  
   - Diseño de CNN sistemático (RegNet) que busca una escalabilidad regular y alto rendimiento.  
   - 32 GFLOPs aproximados y un número considerable de parámetros, superior a ResNet50.

## Entrenamiento

1. **Ejecutar `src/train_model.py`**  

     ```bash
     python src/train_model.py
     ```

   - El script cargará los pesos preentrenados, ajustará la última capa (o más capas descongeladas) y entrenará en tu dataset.  
   - El mejor modelo se guardará en la carpeta `models/`.

## Predicción

- **`app_DragAndDrop.py`**  
  - Carga el modelo guardado.  
  - Lee una imagen nueva, aplica las transformaciones (redimensionar, normalizar).  
  - Predice la clase.

## Resultados de los modelos

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