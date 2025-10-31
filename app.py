import streamlit as st
import pandas as pd
import joblib
import altair as alt
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import requests # Necesario para la descarga

# --- 1. Configuración de la Descarga ---
# ▼▼▼ PEGA AQUÍ EL ID DE TU ARCHIVO DE GOOGLE DRIVE ▼▼▼
GOOGLE_DRIVE_FILE_ID = "https://drive.google.com/file/d/1C33INLaAQi3cJSpwWcnaOc0dDwHEJ2CC/view?usp=sharing"
# ▲▲▲ PEGA AQUÍ EL ID DE TU ARCHIVO DE GOOGLE DRIVE ▲▲▲

MODEL_URL = f"https://drive.google.com/uc?export=download&id=1C33INLaAQi3cJSpwWcnaOc0dDwHEJ2CC"
MODEL_PATH = "modelo_descargado.pkl" # Lo guardamos con este nombre


# --- 2. Función de descarga ---
def download_model(url, file_path):
    """Descarga el modelo si no existe localmente."""
    if not os.path.exists(file_path):
        with st.spinner(f"Descargando el modelo (345 MB)... Esto puede tardar unos minutos la primera vez."):
            try:
                # Usamos requests con stream=True para archivos grandes
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(file_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192): 
                            f.write(chunk)
                st.success("Modelo descargado exitosamente.")
            except Exception as e:
                st.error(f"Error al descargar el modelo: {e}.")
                st.error("Por favor, verifica que el ID de Google Drive sea correcto y que el enlace sea 'Público para cualquier persona'.")
                st.stop()
    else:
        # El modelo ya existe, no hacemos nada
        pass

# --- 3. Configuración de la Página ---
st.set_page_config(
    page_title="Análisis y Predicción de Ingresos de Películas",
    page_icon="🎬",
    # ... (el resto de tu st.set_page_config) ...
)

# --- 4. Carga de Activos (Modelo y Datos) ---
@st.cache_resource
def load_model(file_path):
    """Carga el modelo desde el path local."""
    try:
        model = joblib.load(file_path)
        return model
    except Exception as e:
        st.error(f"Error fatal al cargar el modelo: {e}")
        st.stop()

@st.cache_data
def load_data():
    # ... (tu función load_data sin cambios) ...
    try:
        df = pd.read_csv('tmdb_dataset_accion_2000_presente_10k.csv')
        # ... (resto del preprocesamiento) ...
        return df
    except Exception as e:
        st.error(f"Error fatal al cargar los datos: {e}")
        st.stop()
    
@st.cache_resource
def get_shap_explainer(_model):
    # ... (tu función get_shap_explainer sin cambios) ...
    regressor = _model.named_steps['regressor']
    return shap.TreeExplainer(regressor)

# --- 5. INICIO DE LA APP ---

# --- ¡MUY IMPORTANTE! ---
# 1. Ejecutar la descarga ANTES de cargar el modelo
download_model(MODEL_URL, MODEL_PATH)

# 2. Cargar el modelo (ahora desde el archivo local descargado)
model = load_model(MODEL_PATH)
df_raw = load_data()
shap_explainer = get_shap_explainer(model)

model_features = [
    'score', 'movie_popularity', 'actor1_popularity', 'actor2_popularity', 
    'actor3_popularity', 'budget', 'actor1_age', 'actor2_age', 'actor3_age'
]

# 3. Barra Lateral (Sidebar)
st.sidebar.title("🤖 Probar el Modelo")
# ... (El resto de tu código de la app, st.sidebar.form, st.tabs, etc.) ...
# ... (Pega aquí el resto de tu app.py anterior) ...

# --- CÓDIGO RESTANTE (Pestañas, etc.) ---
# (Asegúrate de pegar aquí el resto de tu código anterior: st.title, st.tabs, 
# la lógica de st.session_state, y el contenido de tab1, tab2, y tab3)
st.title("🎬 Análisis y Predicción de Ingresos de Películas de Acción")
st.markdown("Plataforma interactiva para el **Trabajo Práctico Integrador - Grupo 21**.")

tab1, tab2, tab3 = st.tabs([
    "📊 Hallazgos y Visualizaciones", 
    "📈 Prueba del Modelo (en vivo)", 
    "📄 Sobre el Proyecto"
])

# Lógica de predicción
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

if submit_button:
    # ... (pega tu bloque 'if submit_button:' aquí) ...
    
# Pestaña 1: Visualizaciones
with tab1:
    # ... (pega el contenido de tu 'with tab1:' aquí) ...

# Pestaña 2: Prueba del Modelo
with tab2:
    # ... (pega el contenido de tu 'with tab2:' aquí) ...

# Pestaña 3: Sobre el Proyecto
with tab3:
    # ... (pega el contenido de tu 'with tab3:' aquí) ...