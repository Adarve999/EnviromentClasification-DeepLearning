import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from cnn import *
# =============================================================================
# IMPORTANTE:
# Se asume que ya cuentas con la función load_data definida en tu código, la cual
# debe cargar el dataset organizado en carpetas y devolver:
#    train_loader, valid_loader, num_classes
#
# Ejemplo de firma:
# def load_data(train_dir, valid_dir, batch_size, img_size):
#     # Implementación usando, por ejemplo, torchvision.datasets.ImageFolder
#     return train_loader, valid_loader, num_classes
# =============================================================================

# Modo de la aplicación: Entrenamiento o Clasificación
mode = st.sidebar.selectbox("Selecciona el modo", ["Entrenar Modelo", "Clasificar Imagen"])

if mode == "Entrenar Modelo":
    st.title("Entrenamiento de Modelo con EfficientNet-B7 Fine Tuning")
    
    # Parámetros de entrada para el entrenamiento
    train_dir = st.text_input("Directorio de entrenamiento", value="../../../data/training")
    valid_dir = st.text_input("Directorio de validación", value="../../../data/validation")
    batch_size = st.number_input("Batch size", value=32, step=1)
    epochs = st.number_input("Número de épocas", value=5, step=1)  # Para demo se recomienda pocas épocas
    learning_rate = st.number_input("Tasa de aprendizaje", value=1e-4, format="%.5f")
    # Para EfficientNet-B7 se recomienda usar imágenes de mayor resolución (por ejemplo, 600x600)
    img_size = st.number_input("Tamaño de la imagen (resolución cuadrada)", value=600, step=1)
    
    if st.button("Iniciar Entrenamiento"):
        # Cargar datos (asegúrate de tener implementada la función load_data)
        with st.spinner("Cargando datos..."):
            train_loader, valid_loader, num_classes = load_data(train_dir, valid_dir, batch_size=batch_size, img_size=img_size)
        st.write(f"Se detectaron {num_classes} clases en el dataset.")
        
        # Cargar el modelo preentrenado con EfficientNet-B7 y modificar la capa final
        weights = EfficientNet_B7_Weights.DEFAULT
        model = efficientnet_b7(weights=weights)
        # Reemplazar la capa final para que se ajuste al número de clases de tu dataset
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
        
        # Entrenamiento
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
        # Guardar el modelo y las clases en session_state para usarlos luego en la clasificación
        st.session_state["model"] = model
        # Suponiendo que load_data utiliza ImageFolder, las clases se pueden obtener así:
        st.session_state["classes"] = train_loader.dataset.classes

elif mode == "Clasificar Imagen":
    st.title("Clasificar Imagen con Modelo Fine Tuned")
    
    # Verificar que el modelo entrenado exista
    if "model" not in st.session_state:
        st.warning("Primero debes entrenar el modelo en la sección 'Entrenar Modelo'.")
    else:
        model = st.session_state["model"]
        classes = st.session_state["classes"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Subir imagen para clasificar
        uploaded_file = st.file_uploader("Carga una imagen (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Imagen cargada", use_column_width=True)
            
            # Transformación recomendada por EfficientNet-B7
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
