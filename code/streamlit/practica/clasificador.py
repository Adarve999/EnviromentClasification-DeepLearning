import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Clasificador de Imágenes", layout="wide")

st.title("Clasificador de Imágenes con EfficientNet-B7")
st.write("Arrastra y suelta una imagen para clasificarla.")

# Drag & Drop de la imagen
uploaded_file = st.file_uploader("Carga una imagen (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Cargar y previsualizar la imagen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_column_width=True)
    
    st.write("Procesando imagen y realizando clasificación...")
    
    # Cargar el modelo preentrenado con los pesos por defecto
    weights = EfficientNet_B7_Weights.DEFAULT
    model = efficientnet_b7(weights=weights)
    model.eval()  # Modo evaluación
    
    # Obtener la transformación recomendada por los pesos
    preprocess = weights.transforms()
    
    # Preprocesar la imagen
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Añadir dimensión de batch

    # Clasificar la imagen sin calcular gradientes
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Obtener las 5 etiquetas más probables
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    st.subheader("Resultados de Clasificación:")
    for i in range(top5_prob.size(0)):
        category = weights.meta["categories"][top5_catid[i]]
        prob = top5_prob[i].item() * 100
        st.write(f"**{category}**: {prob:.2f}%")