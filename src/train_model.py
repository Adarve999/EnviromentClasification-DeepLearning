#!/usr/bin/env python
import sys
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from utils.cnn import CNN, load_data

def main():
    available_models = torchvision.models.list_models(module=torchvision.models)
    print("Modelos disponibles:")
    for model_name in available_models:
        print(" -", model_name)

    selected_model = input("\nIngrese el nombre del modelo a utilizar: ").strip()
    if selected_model not in available_models:
        print(f"\nEl modelo '{selected_model}' no se encontró en la lista de modelos disponibles.")
        sys.exit(1)
    
    chosen_model = torchvision.models.__dict__[selected_model](weights="DEFAULT")
    print(f"\nModelo '{selected_model}' cargado correctamente.")

    train_dir = os.path.join("..", "data", "training")
    valid_dir = os.path.join("..", "data", "validation")

    batch_size = 128
    img_size = 224  # (ej. ResNet requiere 224x224)
    
    train_loader, valid_loader, num_classes = load_data(train_dir, valid_dir, batch_size=batch_size, img_size=img_size)
    print(f"\nSe detectaron {num_classes} clases en el dataset.")

    model = CNN(chosen_model, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    try:
        epochs = int(input("\nIngrese el número de épocas para el entrenamiento: "))
    except ValueError:
        print("Valor de época inválido. Se utilizará 1 época por defecto.")
        epochs = 1

    print("\nIniciando entrenamiento...\n")
    model.train_model(train_loader, valid_loader, optimizer, criterion, epochs)

    save_filename = f"{selected_model}-{epochs}epochs"
    model.save_model(save_filename)
    print(f"\nEntrenamiento finalizado. Modelo guardado como '{save_filename}'.")

if __name__ == "__main__":
    main()