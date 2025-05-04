import streamlit as st
import os
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from utils.cnn import CNN, load_data, load_model_weights
import streamlit as st
import base64
from io import BytesIO
from PIL import Image
import glob

# Función para convertir imagen PIL a base64
def image_to_base64(image):
    """Convierte una imagen PIL a base64"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")  # Usa el formato que prefieras (JPEG, PNG, etc.)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Configuración de la página
st.set_page_config(page_title="🧠 Clasificador de Imágenes (Drag & Drop)", layout="wide")

# Cargar y mostrar la imagen de encabezado en la esquina superior derecha
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(CURRENT_DIR, "../image/icaiLogo.jpg")
header_image = Image.open(IMAGE_PATH)

# Usar st.markdown con estilo CSS para posicionar la imagen en la parte superior derecha
st.markdown("""
    <div style="position: absolute; top: 10px; right: 10px;">
        <img src="data:image/jpeg;base64,{}" style="width: 100px; height: auto;" />
    </div>
""".format(image_to_base64(header_image)), unsafe_allow_html=True)


# Título principal
st.markdown("""
    <h1 style='text-align: center; margin-top: 30px; margin-bottom: 30px;'>
        CLASIFICADOR DE IMÁGENES CON IA
    </h1>
""", unsafe_allow_html=True)

st.markdown("""
<div style='border: 2px dotted black; padding: 20px; border-radius: 10px; background-color: transparent;'>
    <h4>📋 Instrucciones:</h4>
    <ol>
        <li>🔍 <strong>Selecciona un modelo</strong> desde la carpeta <code>'models'</code>.</li>
        <li>📂 <strong>Arrastra y suelta</strong> o selecciona una imagen (<code>.jpg</code>, <code>.jpeg</code>, <code>.png</code>).</li>
        <li>🚀 Haz clic en el botón para obtener la <strong>clase predicha</strong>.</li>
    </ol>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<br>
<h4>🏷️ Clases disponibles (con ejemplos visuales):</h4>
<p style="font-size: 15px; color: #444; margin-top: -10px;">
    Los modelos fueron entrenados con imágenes similares a las mostradas abajo. 
    Si subes una imagen muy distinta (por perspectiva, iluminación o contenido), la predicción puede no ser precisa.
</p>
""", unsafe_allow_html=True)

# Muestra de clases con imágenes
classes_dir = os.path.join(CURRENT_DIR, "../data/validation")
class_folders = sorted(os.listdir(classes_dir))

class_translations = {
    "Bedroom": "Dormitorio",
    "Coast": "Costa",
    "Forest": "Bosque",
    "Highway": "Autopista",
    "Industrial": "Zona industrial",
    "Inside city": "Interior urbano",
    "Kitchen": "Cocina",
    "Living room": "Salón",
    "Montain": "Montain",
    "Office": "Oficina",
    "Open country": "Campo abierto",
    "Store": "Tienda",
    "Street": "Calle",
    "Suburb": "Suburbio",
    "Tall building": "Edificio alto"
}

cols = st.columns(5)
for idx, class_name in enumerate(class_folders):
    image_path_pattern = os.path.join(classes_dir, class_name, "*")
    image_files = sorted(glob.glob(image_path_pattern))
    if not image_files:
        continue

    image_path = image_files[0]
    img = Image.open(image_path).convert("RGB")

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    col = cols[idx % 5]
    with col:
        col.markdown(f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <img src="data:image/jpeg;base64,{img_base64}" style="width: 160px; height: 160px; object-fit: cover; border-radius: 10px; border: 1px solid #ccc;"/>
            <div style="margin-top: 5px; font-weight: bold;">{class_name}<br/><span style="font-size: 12px; color: gray;">{class_translations.get(class_name, '')}</span></div>
        </div>
        """, unsafe_allow_html=True)
        
train_dir = os.path.join(CURRENT_DIR, "../data/training")
valid_dir = os.path.join(CURRENT_DIR, '../data/validation')
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
MODELS_DIR = os.path.join(CURRENT_DIR,"../models")
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt") or f.endswith(".pth")]
if not model_files:
    st.error("No se encontraron modelos en la carpeta 'models'. Descargalos aquí: https://drive.google.com/drive/folders/1-9ZGXn6zHftPIvnO7t1JfCHSlHR5p2DT?usp=sharing")
    st.stop()

# Barra lateral para seleccionar el modelo
st.sidebar.header("⚙️ Configuración")
selected_model_file = st.sidebar.selectbox("Selecciona el modelo", model_files)

# Subida de imagen
uploaded_file = st.file_uploader("Selecciona una imagen (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.markdown("<h5 style='text-align: center;'>Vista previa de la imagen</h5>", unsafe_allow_html=True)
    st.image(image, caption="Imagen cargada", width=300)
    gray_image = image.convert("L")
    image = gray_image.convert("RGB")

    st.markdown("---")

    if st.button("Realizar predicción"):
        with st.spinner("Realizando predicción..."):
            # Preprocesamiento
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])
            input_tensor = transform(image).unsqueeze(0)

            # Configurar dispositivo
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.success(f"Usando dispositivo: {device}")

            # Extraer modelo y arquitectura
            model_identifier = os.path.splitext(selected_model_file)[0]
            arch_name = selected_model_file.split("-")[0]
            
            if arch_name not in torchvision.models.list_models(module=torchvision.models):
                st.error(f"Arquitectura no encontrada: '{arch_name}' en torchvision.models.")
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

            # Predicción
            predicted_labels = my_trained_model.predict_ImageOnly(input_tensor)
            predicted_index = predicted_labels[0]
            predicted_label = classnames[predicted_index]

            with torch.no_grad():
                output = my_trained_model(input_tensor)  # Salida de la red
                probabilities = torch.nn.functional.softmax(output, dim=1)  # Convertir a probabilidades
                probs = probabilities.squeeze().cpu().numpy()  # Convertir a numpy para mostrar

            # Mostrar la predicción final
            st.markdown(f"""
            <div style='background-color: #ffffffb3; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50; margin-top: 20px;'>
                <h4 style='color: #2e7d32;'>Resultado de Clasificación:</h4>
                <p style='font-size: 20px;'>La imagen ha sido clasificada como: <strong>{predicted_label}</strong></p>
            </div>
            """, unsafe_allow_html=True)

# Footer fijo
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #f0f2f6;
            color: #6c757d;
            text-align: left;
            padding: 10px;
            font-size: 14px;
            border-top: 1px solid #dee2e6;
            z-index: 9999;
        }
    </style>
    <div class="footer">
        Desarrollado por Marta Rodríguez Hebles, Maria Valvanera Gil de Biedma, Blanca Sayas Ladaga y Rubén Adarve Pérez
    </div>
""", unsafe_allow_html=True)
