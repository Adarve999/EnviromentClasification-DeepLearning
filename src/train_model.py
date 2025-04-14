#!/usr/bin/env python
import sys
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import wandb
import datetime
from utils.cnn import CNN, load_data, load_model_weights
from torch.optim.lr_scheduler import OneCycleLR

def main():
    # Mostrar modelos disponibles en torchvision
    available_models = torchvision.models.list_models(module=torchvision.models)
    print("Modelos disponibles:")
    for model_name in available_models:
        print(" -", model_name)

    # Pedir nombre del modelo
    selected_model = input("\nIngrese el nombre del modelo a utilizar: ").strip()
    if selected_model not in available_models:
        print(f"\nEl modelo '{selected_model}' no se encontró en la lista de modelos disponibles.")
        sys.exit(1)
    
    chosen_model = torchvision.models.__dict__[selected_model](weights="DEFAULT")
    print(f"\nModelo '{selected_model}' cargado correctamente.")

    # Pedir learning rate al usuario, con default 1e-4
    try:
        lr_input = input("Ingrese el learning rate (por defecto 1e-4): ").strip()
        if lr_input == "":
            learning_rate = 1e-4
        else:
            learning_rate = float(lr_input)
    except ValueError:
        print("Valor del learning rate inválido. Se utilizará 1e-4 por defecto.")
        learning_rate = 1e-4
    
    # Pedir número de capas a descongelar, con default 0
    try:
        unfreeze_input = input("Ingrese la cantidad de capas a 'descongelar' (por defecto 0): ").strip()
        if unfreeze_input == "":
            unfreeze_layer = 0
        else:
            unfreeze_layer = int(unfreeze_input)
    except ValueError:
        print("Valor inválido de capas a descongelar. Se utilizará 0 por defecto.")
        unfreeze_layer = 0

    # Definir directorios y parámetros
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(CURRENT_DIR, "../data/training")
    valid_dir = os.path.join(CURRENT_DIR, '../data/validation')
    img_size = 224
    batch_size = 32

    # Cargar datos
    train_loader, valid_loader, num_classes = load_data(
        train_dir, 
        valid_dir, 
        batch_size=batch_size, 
        img_size=img_size
    )
    print(f"\nSe detectaron {num_classes} clases en el dataset.")

    # Crear modelo CNN con unfreeze_layer
    model = CNN(chosen_model, num_classes, unfreezed_layers=unfreeze_layer)
    
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsando dispositivo: {device}")
    model = model.to(device)
    
    # Definir optimizer y loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Pedir número de épocas
    try:
        epochs = int(input("\nIngrese el número de épocas para el entrenamiento: "))
    except ValueError:
        print("Valor de épocas inválido. Se utilizará 1 época por defecto.")
        epochs = 1

    # Inicializar wandb
    run_name = f"{selected_model}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    wandb.init(
        entity="machinelearningicai2025",
        project="Weight&Bias",
        name=run_name,
        config={"model": selected_model, "epochs": epochs, "lr": learning_rate, "unfreeze_layer": unfreeze_layer}
    )

    print("\nIniciando entrenamiento...\n")
    try:
        history = model.train_model(train_loader, valid_loader, optimizer, criterion, epochs, device=device)
    except KeyboardInterrupt:
        print("Entrenamiento cancelado por el usuario. Guardando modelo con los pesos actuales...")
        wandb.log({"message": "Entrenamiento cancelado por el usuario"})
    finally:
        save_filename = f"{selected_model}-{epochs}epochs-Lr{learning_rate}-UL{unfreeze_layer}"
        model.save_model(save_filename)
        print(f"\nModelo guardado como '{save_filename}'.")
        wandb.finish()

if __name__ == "__main__":
    main()