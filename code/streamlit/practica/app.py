import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import os
from cnn import *  # Se asume que load_data está definida en cnn.py

# Definir rutas para guardar el modelo y las clases
MODEL_PATH = "model_efficientnet_b7.pt"
CLASSES_PATH = "classes_efficientnet_b7.pt"

# Modo de la aplicación: Entrenamiento o Clasificación
mode = st.sidebar.selectbox("Selecciona el modo", ["Entrenar Modelo", "Clasificar Imagen"])

if mode == "Entrenar Modelo":
    st.title("Entrenamiento de Modelo con EfficientNet-B7 Fine Tuning")
    
    # Parámetros de entrada para el entrenamiento
    train_dir = st.text_input("Directorio de entrenamiento", value="../../../data/training")
    valid_dir = st.text_input("Directorio de validación", value="../../../data/validation")
    batch_size = st.number_input("Batch size", value=32, step=1)
    epochs = st.number_input("Número de épocas", value=5, step=1) 
    learning_rate = st.number_input("Tasa de aprendizaje", value=1e-4, format="%.5f")
    img_size = st.number_input("Tamaño de la imagen (resolución cuadrada)", value=600, step=1)
    
    # Opción para forzar el reentrenamiento
    retrain = st.checkbox("Forzar reentrenamiento (ignorar modelo existente)", value=False)
    
    if st.button("Iniciar Entrenamiento"):
        if os.path.exists(MODEL_PATH) and not retrain:
            st.info("Modelo previamente entrenado encontrado. Cargando modelo...")
            model = torch.load(MODEL_PATH)
            classes = torch.load(CLASSES_PATH)
            st.session_state["model"] = model
            st.session_state["classes"] = classes
            st.success("Modelo cargado exitosamente.")
        else:
            # Cargar datos usando la función load_data definida en cnn.py
            with st.spinner("Cargando datos..."):
                train_loader, valid_loader, num_classes = load_data(train_dir, valid_dir, batch_size=batch_size, img_size=img_size)
            st.write(f"Se detectaron {num_classes} clases en el dataset.")
            
            # Cargar el modelo preentrenado con EfficientNet-B7 y modificar la capa final
            weights = EfficientNet_B7_Weights.DEFAULT
            model = efficientnet_b7(weights=weights)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
            
            # Congelar la base del modelo y entrenar únicamente la capa final
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
            
            st.write("Iniciando entrenamiento...")
            for epoch in range(1, epochs + 1):
                model.train()
                train_loss = 0.0
                correct = 0
                total = 0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                train_loss /= total
                train_acc = 100 * correct / total
                
                # Validación
                model.eval()
                valid_loss = 0.0
                correct_val = 0
                total_val = 0
                with torch.no_grad():
                    for images, labels in valid_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        valid_loss += loss.item() * images.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        total_val += labels.size(0)
                        correct_val += (predicted == labels).sum().item()
                valid_loss /= total_val
                valid_acc = 100 * correct_val / total_val
                
                st.write(f"Época {epoch}/{epochs}  |  Pérdida Entrenamiento: {train_loss:.4f}  |  Precisión Entrenamiento: {train_acc:.2f}%")
                st.write(f"Pérdida Validación: {valid_loss:.4f}  |  Precisión Validación: {valid_acc:.2f}%")
            
            st.success("Entrenamiento finalizado.")
            # Guardar el modelo y las clases en disco
            torch.save(model, MODEL_PATH)
            torch.save(train_loader.dataset.classes, CLASSES_PATH)
            st.session_state["model"] = model
            st.session_state["classes"] = train_loader.dataset.classes

elif mode == "Clasificar Imagen":
    st.title("Clasificar Imagen con Modelo Fine Tuned")
    
    # Intentar cargar el modelo desde session_state o desde disco
    if "model" not in st.session_state:
        if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
            st.info("Cargando modelo entrenado desde disco...")
            model = torch.load(MODEL_PATH)
            classes = torch.load(CLASSES_PATH)
            st.session_state["model"] = model
            st.session_state["classes"] = classes
        else:
            st.warning("Primero debes entrenar el modelo en la sección 'Entrenar Modelo'.")
    
    if "model" in st.session_state:
        model = st.session_state["model"]
        classes = st.session_state["classes"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        uploaded_file = st.file_uploader("Carga una imagen (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Imagen cargada", use_column_width=True)
            
            # Aplicar la transformación recomendada por EfficientNet-B7
            weights = EfficientNet_B7_Weights.DEFAULT
            preprocess = weights.transforms()
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            
            st.subheader("Resultados de Clasificación:")
            for i in range(top5_prob.size(0)):
                st.write(f"**{classes[top5_catid[i]]}**: {top5_prob[i].item() * 100:.2f}%")
