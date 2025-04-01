import streamlit as st
import os
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from utils.cnn import CNN, load_data, load_model_weights

st.set_page_config(page_title="Clasificador de Imágenes (Drag & Drop)", layout="wide")
st.title("Clasificador de Imágenes - Predicción de una imagen cualquiera")

st.write("""
1. Selecciona un modelo de la carpeta 'models'.
2. Arrastra y suelta (o busca) una imagen (jpg, jpeg, png).
3. Haz clic en el botón para ver la clase predicha.
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

# Carpeta de modelos
MODELS_DIR = "models"
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt") or f.endswith(".pth")]
if not model_files:
    st.error("No se encontraron modelos en la carpeta 'models'.")
    st.stop()

selected_model_file = st.selectbox("Selecciona el modelo", model_files)
uploaded_file = st.file_uploader("Carga una imagen (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", width=300)
    gray_image = image.convert("L")
    image = gray_image.convert("RGB")
    
    if st.button("Realizar predicción"):
        st.write("Realizando predicción...")

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
            weights="IMAGENET1K_SWAG_LINEAR_V1"
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
