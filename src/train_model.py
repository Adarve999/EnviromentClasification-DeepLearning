#!/usr/bin/env python
import sys
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import wandb
import datetime
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
    learning_rate = 1e-4 
    batch_size = 32
    img_size = 224  # Por ejemplo, ResNet requiere 224x224

    train_loader, valid_loader, num_classes = load_data(train_dir, valid_dir, batch_size=batch_size, img_size=img_size)
    print(f"\nSe detectaron {num_classes} clases en el dataset.")

    model = CNN(chosen_model, num_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsando dispositivo: {device}")
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    try:
        epochs = int(input("\nIngrese el número de épocas para el entrenamiento: "))
    except ValueError:
        print("Valor de época inválido. Se utilizará 1 época por defecto.")
        epochs = 1

    run_name = f"{selected_model}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        entity="machinelearningicai2025",
        project="Weight&Bias",
        name=run_name,
        config={"model": selected_model, "epochs": epochs, "lr": learning_rate}
    )

    print("\nIniciando entrenamiento...\n")
    try:
        history = model.train_model(train_loader, valid_loader, optimizer, criterion, epochs, device=device)
    except KeyboardInterrupt:
        print("Entrenamiento cancelado por el usuario. Guardando modelo con los pesos actuales...")
        wandb.log({"message": "Entrenamiento cancelado por el usuario"})
    finally:
        save_filename = f"{selected_model}-{epochs}epochs"
        model.save_model(save_filename)
        print(f"\nModelo guardado como '{save_filename}'.")
        wandb.finish()

if __name__ == "__main__":
    main()
