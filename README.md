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
