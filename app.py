import streamlit as st
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import onnxruntime as ort

st.set_page_config(
    page_title="Chiari I — Herramienta de diagnostico",
    page_icon="🧠",
    layout="centered",
)

@st.cache_resource
def cargar_modelo():
    ruta = Path("models")
    candidatos = sorted(ruta.glob("*.onnx"))
    if not candidatos:
        return None, None
    ruta_modelo = candidatos[0]
    session = ort.InferenceSession(str(ruta_modelo))
    return session, ruta_modelo.stem

def preprocesar(imagen_pil, size=(224, 224)):
    arr = np.array(imagen_pil.convert("L").resize(size, Image.LANCZOS), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr = clahe.apply(arr)
    arr_f = arr.astype(np.float32) / 255.0
    arr_f = np.stack([arr_f, arr_f, arr_f], axis=-1)
    return np.expand_dims(arr_f, 0).astype(np.float32)

# --- Interfaz ---
st.title("🧠 Diagnostico de Malformacion de Arnold-Chiari Tipo I")
st.markdown(
    "Herramienta de apoyo diagnostico basada en redes neuronales convolucionales.  \n"
    "Clasifica MRI sagital T1 como **Normal** o **Chiari I**.  \n"
    "⚠️ *Apoyo academico. No reemplaza el criterio medico.*"
)
st.divider()

with st.spinner("Cargando modelo..."):
    session, nombre_modelo = cargar_modelo()

if session is None:
    st.error("No se encontro modelo .onnx en models/. Ejecuta convert_to_onnx.py primero.")
    st.stop()

st.success(f"Modelo: `{nombre_modelo}`")
st.divider()

st.subheader("Carga una imagen de resonancia magnetica sagital")
archivo = st.file_uploader("Selecciona la imagen", type=["jpg", "jpeg", "png"],
                           label_visibility="collapsed")

if archivo is not None:
    imagen_pil = Image.open(archivo)
    col1, col2 = st.columns(2)

    with col1:
        st.image(imagen_pil, caption="Imagen cargada", use_container_width=True)

    with st.spinner("Analizando..."):
        img_tensor = preprocesar(imagen_pil)
        input_name = session.get_inputs()[0].name
        prob = float(session.run(None, {input_name: img_tensor})[0][0][0])

    prediccion = "Chiari I" if prob >= 0.5 else "Normal"
    with col2:
        if prediccion == "Chiari I":
            st.error(f"### 🔴 {prediccion}")
        else:
            st.success(f"### 🟢 {prediccion}")
        st.metric("Probabilidad Chiari I", f"{prob*100:.1f}%")
        st.progress(float(prob))

    st.divider()
    st.warning(
        "**Advertencia:** Herramienta academica de proyecto de grado. "
        "El diagnostico definitivo debe realizarlo un medico especialista."
    )

st.divider()
st.caption(f"Proyecto de grado | Clasificacion Chiari Tipo I | Modelo: {nombre_modelo}")
