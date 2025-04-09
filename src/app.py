import streamlit as st
import os
import torch
import torchvision
import random
from PIL import Image
from torchvision import transforms
from utils.cnn import CNN, load_data, load_model_weights

st.set_page_config(page_title="Clasificador de Imágenes (Validación Aleatoria)", layout="wide")
st.title("Clasificador de Imágenes - Predicción de UNA sola imagen aleatoria del conjunto de validación")

st.write("""
Este ejemplo toma **una sola** imagen aleatoria del conjunto de validación y muestra la clase que el modelo predice para esa imagen,
sin predecir todo el dataset.
""")

train_dir = '../data/training'
valid_dir = '../data/validation'
batch_size = 32
img_size = 224

train_loader, valid_loader, num_classes = load_data(
    train_dir, 
    valid_dir, 
    batch_size=batch_size, 
    img_size=img_size
)
classnames = train_loader.dataset.classes

MODELS_DIR = "models"
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt") or f.endswith(".pth")]
if not model_files:
    st.error("No se encontraron modelos en la carpeta 'models'.")
    st.stop()

selected_model_file = st.selectbox("Selecciona el modelo", model_files)

if st.button("Obtener imagen aleatoria y predecir"):
    valid_samples = valid_loader.dataset.samples
    rand_index = random.choice(range(len(valid_samples)))
    image_path, _ = valid_samples[rand_index]
    
    image = Image.open(image_path).convert("RGB")
    st.image(image, caption=f"Imagen aleatoria: {os.path.basename(image_path)}", width=300)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Usando dispositivo: {device}")

    model_identifier = os.path.splitext(selected_model_file)[0]
    arch_name = selected_model_file.split("-")[0]
    if arch_name not in torchvision.models.list_models(module=torchvision.models):
        st.error(f"No se encontró la arquitectura '{arch_name}' en torchvision.models.")
        st.stop()

    if("regnet" in model_identifier):
        weights="IMAGENET1K_SWAG_E2E_V1"
    else:
        weights="DEFAULT"

    base_model = torchvision.models.__dict__[arch_name](weights=weights)
    my_trained_model = CNN(base_model, num_classes)
    my_trained_model.to(device)

    model_weights = load_model_weights(model_identifier, device=device)
    my_trained_model.load_state_dict(model_weights)
    my_trained_model.eval()

    input_tensor = input_tensor.to(device)
    predicted_labels = my_trained_model.predict_ImageOnly(input_tensor)
    predicted_index = predicted_labels[0]
    predicted_label = classnames[predicted_index]

    st.subheader("Resultado de Clasificación:")
    st.write(f"La imagen ha sido clasificada como: **{predicted_label}**")
