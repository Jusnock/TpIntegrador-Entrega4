import streamlit as st
import pandas as pd
import joblib
import altair as alt
import numpy as np

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Predicción de Ingresos de Películas",
    page_icon="🎬",
    layout="wide"
)

# --- Carga de Activos (Modelo y Datos) ---
# Usamos cache para que no se carguen cada vez que interactuamos
@st.cache_resource
def load_model():
    """Carga el modelo CatBoost entrenado."""
    try:
        model = joblib.load('best_catboost_model_optimized_es_winsorized.pkl')
        return model
    except FileNotFoundError:
        st.error("Error: Archivo 'best_catboost_model_optimized_es_winsorized.pkl' no encontrado.")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.stop()

@st.cache_data
def load_data():
    """Carga el dataset de películas."""
    try:
        # Usamos el mismo preprocesamiento básico del notebook para la visualización
        df = pd.read_csv('tmdb_dataset_accion_2000_presente_10k.csv')
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['profit'] = df['revenue'] - df['budget']
        df['profit_percentage'] = ((df['profit'] / df['budget']) * 100).replace([np.inf, -np.inf], np.nan)
        return df
    except FileNotFoundError:
        st.error("Error: Archivo 'tmdb_dataset_accion_2000_presente_10k.csv' no encontrado.")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        st.stop()

# Cargar modelo y datos
model = load_model()
df_raw = load_data()

# Obtener la lista de columnas que espera el modelo (basado en tu notebook)
# 'revenue' se excluye porque es el objetivo, pero 'budget' sí está
model_features = [
    'score', 
    'movie_popularity', 
    'actor1_popularity', 
    'actor2_popularity', 
    'actor3_popularity', 
    'budget', 
    'actor1_age', 
    'actor2_age', 
    'actor3_age'
]


# --- Navegación de la App (Sidebar) ---
st.sidebar.title("🎬 Navegación")
page = st.sidebar.radio("Selecciona una página:", ["Predicción de Ingresos", "Exploración de Datos"])


# =============================================================================
# --- PÁGINA 1: PREDICCIÓN DE INGRESOS ---
# =============================================================================

if page == "Predicción de Ingresos":
    
    st.title("🤖 Predice el Éxito de una Película")
    st.write("Ingresa los datos de una película para obtener una predicción de sus ingresos (revenue).")
    st.write("El modelo fue entrenado en un dataset de ~10,000 películas de acción (2000-Presente).")
    
    # --- Formulario de Inputs del Usuario ---
    st.sidebar.header("Ingresa los Datos de la Película:")
    
    # Creamos sliders y number_inputs en la barra lateral
    budget = st.sidebar.number_input("Presupuesto (Budget)", min_value=1000000, max_value=400000000, value=50000000, step=1000000, format="%d")
    score = st.sidebar.slider("Puntaje TMDB (Score)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    movie_popularity = st.sidebar.slider("Popularidad de la Película", min_value=10.0, max_value=500.0, value=100.0, step=5.0)
    
    st.sidebar.subheader("Actores")
    actor1_popularity = st.sidebar.slider("Popularidad Actor 1", min_value=1.0, max_value=100.0, value=15.0)
    actor1_age = st.sidebar.slider("Edad Actor 1", min_value=18, max_value=80, value=45)
    
    actor2_popularity = st.sidebar.slider("Popularidad Actor 2", min_value=1.0, max_value=80.0, value=10.0)
    actor2_age = st.sidebar.slider("Edad Actor 2", min_value=18, max_value=80, value=40)
    
    actor3_popularity = st.sidebar.slider("Popularidad Actor 3", min_value=1.0, max_value=60.0, value=5.0)
    actor3_age = st.sidebar.slider("Edad Actor 3", min_value=18, max_value=80, value=35)

    # --- Lógica de Predicción ---
    if st.button("📈 Predecir Ingresos", type="primary"):
        
        # 1. Crear el DataFrame de entrada
        # (Debe tener EXACTAMENTE las mismas columnas que 'X_train_new_winsorized' en tu notebook)
        input_data = {
            'score': [score],
            'movie_popularity': [movie_popularity],
            'actor1_popularity': [actor1_popularity],
            'actor2_popularity': [actor2_popularity],
            'actor3_popularity': [actor3_popularity],
            'budget': [float(budget)], # Asegurarse que sea float
            'actor1_age': [float(actor1_age)],
            'actor2_age': [float(actor2_age)],
            'actor3_age': [float(actor3_age)]
        }
        
        # Reordenar columnas para que coincidan con el entrenamiento
        input_df = pd.DataFrame(input_data)[model_features] 

        # 2. Realizar la predicción
        # Tu notebook alimenta datos "raw" al pipeline, así que replicamos eso.
        prediction = model.predict(input_df)
        
        # 3. Mostrar resultados
        predicted_revenue = prediction[0]
        profit = predicted_revenue - budget
        profit_percent = (profit / budget) * 100

        st.subheader("Resultados de la Predicción:")
        col1, col2 = st.columns(2)
        
        col1.metric(
            label="Ingreso (Revenue) Predicho", 
            value=f"${predicted_revenue:,.0f}"
        )
        col2.metric(
            label="Ganancia/Pérdida Estimada",
            value=f"${profit:,.0f}",
            delta=f"{profit_percent:.2f} %"
        )
        
        st.info("Nota: Esta es una predicción basada en un modelo de Machine Learning ($R^2 \approx 0.79$) y no garantiza resultados reales.", icon="💡")

# =============================================================================
# --- PÁGINA 2: EXPLORACIÓN DE DATOS (CON ALTAIR) ---
# =============================================================================

elif page == "Exploración de Datos":
    st.title("📊 Exploración Interactiva de Datos")
    st.write("Visualiza las relaciones en el dataset de películas de acción.")

    # --- Filtro de Datos ---
    st.subheader("Filtros del Dataset")
    # Usamos un 'sample' para que los gráficos sean más rápidos
    df_sample = df_raw.sample(n=2000, random_state=42).copy()

    # Filtro de presupuesto
    budget_range = st.slider(
        "Filtrar por Presupuesto (Budget)",
        min_value=float(df_sample['budget'].min()),
        max_value=float(df_sample['budget'].max()),
        value=(float(df_sample['budget'].min()), float(df_sample['budget'].max()))
    )
    df_filtered = df_sample[
        (df_sample['budget'] >= budget_range[0]) & 
        (df_sample['budget'] <= budget_range[1])
    ]

    st.write(f"Mostrando {len(df_filtered)} películas (de 2000 aleatorias).")

    # --- Visualizaciones de Altair ---
    
    st.header("Gráfico 1: Presupuesto vs. Ingresos (Requerido)")
    st.write("Relación entre el costo de una película y sus ingresos en taquilla.")
    
    # Gráfico de dispersión interactivo con Altair
    scatter_budget_revenue = alt.Chart(df_filtered).mark_circle(opacity=0.7).encode(
        # Eje X: Presupuesto
        x=alt.X('budget:Q', title='Presupuesto ($)', axis=alt.Axis(format='$,.0f')),
        
        # Eje Y: Ingresos
        y=alt.Y('revenue:Q', title='Ingresos ($)', axis=alt.Axis(format='$,.0f')),
        
        # Color basado en el puntaje
        color=alt.Color('score:Q', title='Puntaje TMDB', scale=alt.Scale(range='heatmap')),
        
        # Tooltip para detalles al pasar el mouse
        tooltip=[
            alt.Tooltip('title', title='Título'),
            alt.Tooltip('budget', title='Presupuesto', format='$,.0f'),
            alt.Tooltip('revenue', title='Ingresos', format='$,.0f'),
            alt.Tooltip('score', title='Puntaje')
        ]
    ).properties(
        title='Presupuesto vs. Ingresos por Puntaje'
    ).interactive() # <-- Esto lo hace interactivo (zoom y pan)
    
    st.altair_chart(scatter_budget_revenue, use_container_width=True)

    # --- Aquí puedes agregar tus otras 2 visualizaciones ---
    
    st.header("Gráfico 2: Popularidad vs. Puntaje")
    st.write("¿Las películas más populares tienen mejores puntajes?")
    
    scatter_pop_score = alt.Chart(df_filtered).mark_point(filled=True, size=60).encode(
        x=alt.X('movie_popularity:Q', title='Popularidad', scale=alt.Scale(type='log')), # Escala log para popularidad
        y=alt.Y('score:Q', title='Puntaje (Score)', scale=alt.Scale(zero=False)),
        color=alt.Color('profit_percentage:Q', 
                        title='% Ganancia', 
                        scale=alt.Scale(type='log', domainMid=0, range='diverging')),
        tooltip=[
            alt.Tooltip('title', title='Título'),
            alt.Tooltip('movie_popularity', title='Popularidad'),
            alt.Tooltip('score', title='Puntaje'),
            alt.Tooltip('profit_percentage', title='% Ganancia', format='.1f')
        ]
    ).properties(
        title='Popularidad (Log) vs. Puntaje, coloreado por % de Ganancia'
    ).interactive()

    st.altair_chart(scatter_pop_score, use_container_width=True)


    st.header("Gráfico 3: Distribución de Ingresos")
    st.write("Histograma que muestra cuántas películas caen en diferentes rangos de ingresos.")
    
    histogram_revenue = alt.Chart(df_filtered).mark_bar().encode(
        x=alt.X('revenue:Q', bin=alt.Bin(maxbins=40), title='Ingresos ($)'),
        y=alt.Y('count()', title='Cantidad de Películas'),
        tooltip=[
            alt.Tooltip('revenue:Q', bin=True, title='Rango de Ingresos'),
            alt.Tooltip('count()', title='Cantidad de Películas')
        ]
    ).properties(
        title='Distribución de Ingresos en Taquilla'
    ).interactive()
    
    st.altair_chart(histogram_revenue, use_container_width=True)