# machineLearning_II

## instalar dependencias para usar GPU

Instalar dependencias necesarias:

```
conda install pytorch torchvision torchaudio cudatoolkit -c pytorch
conda install -c conda-forge cudatoolkit
conda install -c conda-forge cudnn
```

validar codigo si se está utilizando cuda:

```
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.enabled)
print(torch.cuda.is_available())
```
## Modelos probados

- Resnet50
- resnext101_32x8d
- resnext101_64x4d
- regnet_y_32gf
- regnet_y_128gf
