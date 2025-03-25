import streamlit as st
import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from utils.cnn import CNN, load_model_weights

st.set_page_config(page_title="Clasificador de Imágenes", layout="wide")
st.title("Clasificador de Imágenes - Selección de Modelo")
st.write("Carga una imagen y selecciona un modelo entrenado para clasificarla.")

# Carpeta donde se encuentran los modelos guardados (archivos .pt o .pth con state_dict)
MODELS_DIR = "models"
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt") or f.endswith(".pth")]

if not model_files:
    st.error("No se encontraron modelos en la carpeta 'models'.")
    st.stop()

# Dropdown para seleccionar el modelo
selected_model_file = st.selectbox("Selecciona el modelo", model_files)

# Se extrae el identificador del modelo sin extensión
model_identifier = os.path.splitext(selected_model_file)[0]

# Definir las clases personalizadas (en este caso, 15 clases)
custom_classes = [
    'Bedroom',
    'Coast',
    'Forest',
    'Highway',
    'Industrial',
    'Inside city',
    'Kitchen',
    'Living room',
    'Mountain',
    'Office',
    'Open country',
    'Store',
    'Street',
    'Suburb',
    'Tall building'
]
num_classes = len(custom_classes)

# Drag & Drop para cargar la imagen
uploaded_file = st.file_uploader("Carga una imagen (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Cargar la imagen
    image = Image.open(uploaded_file).convert("RGB")
    # Previsualización de tamaño reducido (por ejemplo, ancho=300 px)
    st.image(image, caption="Imagen cargada", width=300)

    # Botón para iniciar la predicción
    if st.button("Realizar predicción"):
        st.write("Realizando predicción...")

        # Preprocesar la imagen: redimensionar a 224x224 y normalizar según ImageNet
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)  # Añadir dimensión de batch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.write(f"Usando dispositivo: {device}")

        arch_name = selected_model_file.split("-")[0]
        if arch_name not in torchvision.models.list_models(module=torchvision.models):
            st.error(f"No se encontró la arquitectura '{arch_name}' en torchvision.models.")
            st.stop()

        base_model = torchvision.models.__dict__[arch_name](weights="DEFAULT")
        my_trained_model = CNN(base_model, num_classes)
        my_trained_model.to(device)

        model_weights = load_model_weights(model_identifier)
        my_trained_model.load_state_dict(model_weights)
        my_trained_model.eval()

        with torch.no_grad():
            output = my_trained_model(input_batch.to(device))
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        st.subheader("Resultados de Clasificación:")
        for i in range(top5_prob.size(0)):
            if top5_catid[i] < num_classes:
                label = custom_classes[top5_catid[i]]
            else:
                label = f"Clase desconocida ({top5_catid[i].item()})"
            st.write(f"**{label}**: {top5_prob[i].item()*100:.2f}%")