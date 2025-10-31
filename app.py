import streamlit as st
import pandas as pd
import joblib
import altair as alt
import numpy as np

# --- 1. Configuración de la Página (Más Profesional) ---
st.set_page_config(
    page_title="Análisis y Predicción de Ingresos de Películas",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/jusnock/tpintegrador-entrega4',
        'Report a bug': 'https://github.com/jusnock/tpintegrador-entrega4/issues',
        'About': """
        ## Aplicación para el Trabajo Práctico Integrador
        
        **Grupo 21 - Cuarta Entrega**
        ... (el resto de tu texto 'About') ...
        """
    }
)

# --- 2. Carga de Activos (Modelo y Datos) ---
# (Sin cambios, el cacheo es eficiente)
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_catboost_model_optimized_es_winsorizado.pkl')
        return model
    except Exception as e:
        st.error(f"Error fatal al cargar el modelo: {e}")
        st.stop()

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('tmdb_dataset_accion_2000_presente_10k.csv')
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['profit'] = df['revenue'] - df['budget']
        # Corregir división por cero si budget es 0
        df['profit_percentage'] = 0.0
        mask = df['budget'] > 0
        df.loc[mask, 'profit_percentage'] = ((df.loc[mask, 'profit'] / df.loc[mask, 'budget']) * 100)
        df['profit_percentage'] = df['profit_percentage'].replace([np.inf, -np.inf], np.nan)
        return df
    except Exception as e:
        st.error(f"Error fatal al cargar los datos: {e}")
        st.stop()

model = load_model()
df_raw = load_data()

model_features = [
    'score', 'movie_popularity', 'actor1_popularity', 'actor2_popularity', 
    'actor3_popularity', 'budget', 'actor1_age', 'actor2_age', 'actor3_age'
]

# --- 3. Barra Lateral (Sidebar) ---
# La barra lateral ahora se usará SOLO para el formulario de predicción.
st.sidebar.title("🤖 Probar el Modelo")
st.sidebar.write("Ingresa los datos para una nueva película y obtén una predicción.")

# --- 4. MEJORA: Usar st.form para la predicción ---
# Esto evita que la app se recargue con cada slider.
# La predicción solo se ejecuta al presionar el botón.
with st.sidebar.form(key="prediction_form"):
    
    st.header("Datos de la Película")
    budget = st.number_input("Presupuesto (Budget)", min_value=1000000, max_value=400000000, value=50000000, step=1000000, format="%d")
    score = st.slider("Puntaje TMDB (Score)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    movie_popularity = st.slider("Popularidad de la Película", min_value=10.0, max_value=500.0, value=100.0, step=5.0)
    
    st.header("Datos de Actores")
    actor1_popularity = st.slider("Popularidad Actor 1", min_value=1.0, max_value=100.0, value=15.0)
    actor1_age = st.slider("Edad Actor 1", min_value=18, max_value=80, value=45)
    
    actor2_popularity = st.slider("Popularidad Actor 2", min_value=1.0, max_value=80.0, value=10.0)
    actor2_age = st.slider("Edad Actor 2", min_value=18, max_value=80, value=40)
    
    actor3_popularity = st.slider("Popularidad Actor 3", min_value=1.0, max_value=60.0, value=5.0)
    actor3_age = st.slider("Edad Actor 3", min_value=18, max_value=80, value=35)

    # El botón que envía el formulario
    submit_button = st.form_submit_button(label="📈 Predecir Ingresos", type="primary", use_container_width=True)


# --- 5. Lógica de Predicción (se ejecuta si se presionó el botón) ---
if submit_button:
    # 1. Crear el DataFrame de entrada
    input_data = {
        'score': [score],
        'movie_popularity': [movie_popularity],
        'actor1_popularity': [actor1_popularity],
        'actor2_popularity': [actor2_popularity],
        'actor3_popularity': [actor3_popularity],
        'budget': [float(budget)], 
        'actor1_age': [float(actor1_age)],
        'actor2_age': [float(actor2_age)],
        'actor3_age': [float(actor3_age)]
    }
    input_df = pd.DataFrame(input_data)[model_features] 

    # 2. Realizar la predicción
    prediction = model.predict(input_df)
    predicted_revenue = prediction[0]
    profit = predicted_revenue - budget
    profit_percent = (profit / budget) * 100

    # 3. Mostrar resultados en la barra lateral
    st.sidebar.subheader("Resultados de la Predicción:")
    st.sidebar.metric(
        label="Ingreso (Revenue) Predicho", 
        value=f"${predicted_revenue:,.0f}"
    )
    st.sidebar.metric(
        label="Ganancia/Pérdida Estimada",
        value=f"${profit:,.0f}",
        delta=f"{profit_percent:.2f} %"
    )
    st.sidebar.info("Nota: Esta predicción se basa en un modelo ($R^2 \approx 0.79$) y no garantiza resultados reales.", icon="💡")


# --- 6. MEJORA: Título Principal y Pestañas (Tabs) ---
# Usar pestañas (st.tabs) es más moderno que el radio button.

st.title("🎬 Análisis y Predicción de Ingresos de Películas de Acción")
st.markdown("Plataforma interactiva para el **Trabajo Práctico Integrador - Grupo 21**.")

tab1, tab2, tab3 = st.tabs([
    "📊 Hallazgos y Visualizaciones", 
    "📈 Prueba del Modelo (en vivo)", 
    "📄 Sobre el Proyecto"
])

# --- Pestaña 1: Visualizaciones ---
with tab1:
    st.header("Exploración Interactiva de los Datos")
    st.write("Visualiza las relaciones clave en el dataset de películas de acción (2000-Presente).")
    
    # Filtro de Datos
    st.subheader("Filtros del Dataset")
    df_sample = df_raw.sample(n=2000, random_state=42).copy()
    budget_range = st.slider(
        "Filtrar por Presupuesto (Budget)",
        min_value=float(df_sample['budget'].min()),
        max_value=float(df_sample['budget'].max()),
        value=(float(df_sample['budget'].min()), float(df_sample['budget'].max()))
    )
    df_filtered = df_sample[
        (df_sample['budget'] >= budget_range[0]) & 
        (df_sample['budget'] <= budget_range[1]) &
        (df_sample['revenue'] > 0) # Asegurar que revenue no sea 0 para logs
    ]
    st.write(f"Mostrando {len(df_filtered)} películas (de 2000 aleatorias).")

    # --- Gráfico 1 (MEJORADO) ---
    st.subheader("Gráfico 1: Presupuesto vs. Ingresos (con Línea de Rentabilidad)")
    
    # Crear la línea de rentabilidad (donde revenue == budget)
    line_data = pd.DataFrame({
        'budget': [df_filtered['budget'].min(), df_filtered['budget'].max()],
        'revenue': [df_filtered['budget'].min(), df_filtered['budget'].max()]
    })
    profit_line = alt.Chart(line_data).mark_line(color='red', strokeDash=[5,5]).encode(
        x='budget:Q',
        y='revenue:Q'
    )
    
    # Gráfico de dispersión
    scatter_budget_revenue = alt.Chart(df_filtered).mark_circle(opacity=0.6).encode(
        x=alt.X('budget:Q', title='Presupuesto ($)', axis=alt.Axis(format='$,.0f')),
        y=alt.Y('revenue:Q', title='Ingresos ($)', axis=alt.Axis(format='$,.0f')),
        color=alt.Color('score:Q', title='Puntaje TMDB', scale=alt.Scale(range='heatmap')),
        size=alt.Size('profit_percentage:Q', title='% Ganancia', legend=alt.Legend(format='.0f')),
        tooltip=[
            alt.Tooltip('title', title='Título'),
            alt.Tooltip('budget', title='Presupuesto', format='$,.0f'),
            alt.Tooltip('revenue', title='Ingresos', format='$,.0f'),
            alt.Tooltip('score', title='Puntaje'),
            alt.Tooltip('profit_percentage', title='% Ganancia', format='.1f')
        ]
    ).interactive() # <-- Interactivo (zoom y pan)
    
    # Combinar gráfico de dispersión y línea
    final_chart_1 = scatter_budget_revenue + profit_line
    st.altair_chart(final_chart_1, use_container_width=True)
    
    # MEJORA: Añadir Hallazgos
    st.markdown("""
    **Hallazgos Clave (Gráfico 1):**
    * **Línea Roja:** Esta es la "línea de rentabilidad" (Presupuesto = Ingresos). Las películas **por encima** de la línea fueron rentables; las que están **por debajo** perdieron dinero.
    * **Color:** El color (de azul a rojo) representa el puntaje de la película. Se observa que muchas películas de alto puntaje (más rojas) se sitúan muy por encima de la línea de rentabilidad.
    * **Tamaño:** El tamaño de la burbuja es el porcentaje de ganancia. Vemos algunas películas de bajo presupuesto con enormes retornos porcentuales.
    """)
    st.divider()

    # --- Gráfico 2 ---
    st.subheader("Gráfico 2: Popularidad (Log) vs. Puntaje")
    
    scatter_pop_score = alt.Chart(df_filtered).mark_point(filled=True, size=60, opacity=0.7).encode(
        x=alt.X('movie_popularity:Q', title='Popularidad (Escala Log)', scale=alt.Scale(type='log')),
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
    st.markdown("""
    **Hallazgos Clave (Gráfico 2):**
    * No hay una correlación clara entre *popularidad* y *puntaje*. Hay películas muy populares con puntajes bajos y viceversa.
    * El color (azul = pérdida, rojo = ganancia) muestra que la rentabilidad está presente en todos los niveles de popularidad y puntaje.
    * Sin embargo, las películas con puntajes muy bajos (ej. < 5.0) tienden a tener más puntos azules (pérdidas).
    """)
    st.divider()

    # --- Gráfico 3 ---
    st.subheader("Gráfico 3: Distribución de Ingresos")
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
    st.markdown("""
    **Hallazgos Clave (Gráfico 3):**
    * La gran mayoría de las películas de acción recaudan menos de $250 millones.
    * El dataset tiene una fuerte "cola larga" (long tail), con unas pocas películas (los "blockbusters") que recaudan cantidades extremadamente altas (ej. +$750M).
    """)

# --- Pestaña 2: Prueba del Modelo ---
with tab2:
    st.header("Prueba del Modelo en Vivo")
    st.write("Usa el formulario en la **barra lateral izquierda** para ingresar los datos de una película.")
    st.image("https://i.imgur.com/gY8DUaB.png", caption="Los resultados de la predicción aparecerán en la barra lateral una vez que presiones el botón.")
    
    st.subheader("Características (Features) Utilizadas por el Modelo")
    st.write("El modelo CatBoost fue entrenado usando estas 9 características de tu notebook:")
    st.code(
        "\n".join(model_features), 
        language="python"
    )

# --- Pestaña 3: Sobre el Proyecto ---
with tab3:
    st.header("Detalles del Trabajo Práctico Integrador")
    
    st.subheader("Grupo 21")
    st.markdown("""
    * **Integrantes:**
    * *<Nombre Apellido 1>*
    * *<Nombre Apellido 2>*
    * *<... (completa con tu grupo)>*
    """)
    
    st.subheader("Contexto del Proyecto")
    st.markdown("""
    Esta aplicación web es la **Cuarta Entrega** del Trabajo Práctico Integrador. El objetivo era construir un pipeline de datos completo, desde la recolección (ETL) y análisis exploratorio (EDA), hasta el entrenamiento de un modelo de Machine Learning y su despliegue en una aplicación interactiva.
    """)
    
    st.subheader("Fuente de Datos")
    st.markdown("""
    Los datos se obtuvieron de **The Movie Database (TMDB)**. Se creó un dataset personalizado de aproximadamente 10,000 películas de acción estrenadas desde el año 2000 hasta la actualidad.
    """)
    
    st.subheader("Modelo de Predicción")
    st.markdown(f"""
    Se entrenó un modelo de regresión **CatBoost** para predecir la variable `revenue` (ingresos). Las 9 características (features) utilizadas se muestran en la pestaña "Prueba del Modelo". 
    
    El modelo final (después de la optimización y el tratamiento de outliers) alcanzó un **$R^2 \approx 0.79$** en el conjunto de prueba.
    """)
    
    st.subheader("Repositorio del Proyecto")
    st.markdown("El código fuente de esta aplicación y de los notebooks de análisis se encuentra en [GitHub](https://github.com/jusnock/tpintegrador-entrega4).") # Reemplaza con tu repo