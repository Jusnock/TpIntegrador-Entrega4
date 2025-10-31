import streamlit as st
import pandas as pd
import joblib
import altair as alt
import numpy as np
import shap  # --- NUEVO ---
import matplotlib.pyplot as plt # --- NUEVO ---

# --- 1. Configuración de la Página ---
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
        * **Objetivo:** Explorar datos y predecir ingresos de películas.
        * **Modelo:** CatBoost Regressor ($R^2 \approx 0.79$).
        """
    }
)

# --- 2. Carga de Activos (Modelo y Datos) ---
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
        df['profit_percentage'] = 0.0
        mask = df['budget'] > 0
        df.loc[mask, 'profit_percentage'] = ((df.loc[mask, 'profit'] / df.loc[mask, 'budget']) * 100)
        df['profit_percentage'] = df['profit_percentage'].replace([np.inf, -np.inf], np.nan)
        return df
    except Exception as e:
        st.error(f"Error fatal al cargar los datos: {e}")
        st.stop()

# --- NUEVO: Cachear el Explainer de SHAP ---
@st.cache_resource
def get_shap_explainer(_model):
    """Crea y cachea el explainer de SHAP para el modelo de árbol."""
    # Extraemos el regresor CatBoost de dentro del Pipeline
    regressor = _model.named_steps['regressor']
    # Usamos TreeExplainer, que es rápido y exacto para CatBoost
    return shap.TreeExplainer(regressor)

model = load_model()
df_raw = load_data()
shap_explainer = get_shap_explainer(model) # Creamos el explainer

model_features = [
    'score', 'movie_popularity', 'actor1_popularity', 'actor2_popularity', 
    'actor3_popularity', 'budget', 'actor1_age', 'actor2_age', 'actor3_age'
]

# --- 3. Barra Lateral (Sidebar) ---
st.sidebar.title("🤖 Probar el Modelo")
st.sidebar.write("Ingresa los datos para una nueva película y obtén una predicción.")

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

    submit_button = st.form_submit_button(label="📈 Predecir Ingresos", type="primary", use_container_width=True)


# --- 4. Título Principal y Pestañas (Tabs) ---
st.title("🎬 Análisis y Predicción de Ingresos de Películas de Acción")
st.markdown("Plataforma interactiva para el **Trabajo Práctico Integrador - Grupo 21**.")

tab1, tab2, tab3 = st.tabs([
    "📊 Hallazgos y Visualizaciones", 
    "📈 Prueba del Modelo (en vivo)", 
    "📄 Sobre el Proyecto"
])

# --- 5. Lógica de Predicción (se ejecuta si se presionó el botón) ---
# --- MEJORA: Usar st.session_state para pasar datos del form a la pestaña ---
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

if submit_button:
    input_data = {
        'score': [score], 'movie_popularity': [movie_popularity], 'actor1_popularity': [actor1_popularity],
        'actor2_popularity': [actor2_popularity], 'actor3_popularity': [actor3_popularity],
        'budget': [float(budget)], 'actor1_age': [float(actor1_age)], 
        'actor2_age': [float(actor2_age)], 'actor3_age': [float(actor3_age)]
    }
    input_df = pd.DataFrame(input_data)[model_features] 

    prediction = model.predict(input_df)
    predicted_revenue = prediction[0]
    profit = predicted_revenue - budget
    profit_percent = (profit / budget) * 100

    # Guardar en el estado de la sesión para mostrar en la Pestaña 2
    st.session_state.prediction_made = True
    st.session_state.predicted_revenue = predicted_revenue
    st.session_state.profit = profit
    st.session_state.profit_percent = profit_percent
    st.session_state.input_df = input_df # Guardamos los inputs para SHAP
    st.session_state.budget = budget


# --- Pestaña 1: Visualizaciones ---
with tab1:
    st.header("Exploración Interactiva de los Datos")
    # (El código de los 3 gráficos de Altair va aquí, sin cambios)
    # ... (pega aquí tus 3 st.subheader y st.altair_chart de la versión anterior) ...
    # --- Por brevedad, no lo repito, pero debe estar aquí ---

    # --- INICIO GRÁFICOS (Pegar código anterior) ---
    st.subheader("Filtros del Dataset")
    df_sample = df_raw.sample(n=2000, random_state=42).copy()
    budget_range = st.slider(
        "Filtrar por Presupuesto (Budget)",
        min_value=float(df_sample['budget'].min()),
        max_value=float(df_sample['budget'].max()),
        value=(float(df_sample['budget'].min()), float(df_sample['budget'].max())),
        key="budget_slider_tab1" # Clave única
    )
    df_filtered = df_sample[
        (df_sample['budget'] >= budget_range[0]) & 
        (df_sample['budget'] <= budget_range[1]) &
        (df_sample['revenue'] > 0)
    ]
    st.write(f"Mostrando {len(df_filtered)} películas (de 2000 aleatorias).")

    # Gráfico 1
    st.subheader("Gráfico 1: Presupuesto vs. Ingresos (con Línea de Rentabilidad)")
    line_data = pd.DataFrame({'budget': [df_filtered['budget'].min(), df_filtered['budget'].max()], 'revenue': [df_filtered['budget'].min(), df_filtered['budget'].max()]})
    profit_line = alt.Chart(line_data).mark_line(color='red', strokeDash=[5,5]).encode(x='budget:Q', y='revenue:Q')
    scatter_budget_revenue = alt.Chart(df_filtered).mark_circle(opacity=0.6).encode(
        x=alt.X('budget:Q', title='Presupuesto ($)', axis=alt.Axis(format='$,.0f')),
        y=alt.Y('revenue:Q', title='Ingresos ($)', axis=alt.Axis(format='$,.0f')),
        color=alt.Color('score:Q', title='Puntaje TMDB', scale=alt.Scale(range='heatmap')),
        size=alt.Size('profit_percentage:Q', title='% Ganancia', legend=alt.Legend(format='.0f')),
        tooltip=['title', alt.Tooltip('budget', format='$,.0f'), alt.Tooltip('revenue', format='$,.0f'), 'score', alt.Tooltip('profit_percentage', format='.1f')]
    ).interactive()
    final_chart_1 = scatter_budget_revenue + profit_line
    st.altair_chart(final_chart_1, use_container_width=True)
    st.markdown("**Hallazgos Clave (Gráfico 1):** ... (pega tus hallazgos aquí) ...")
    st.divider()

    # Gráfico 2
    st.subheader("Gráfico 2: Popularidad (Log) vs. Puntaje")
    scatter_pop_score = alt.Chart(df_filtered).mark_point(filled=True, size=60, opacity=0.7).encode(
        x=alt.X('movie_popularity:Q', title='Popularidad (Escala Log)', scale=alt.Scale(type='log')),
        y=alt.Y('score:Q', title='Puntaje (Score)', scale=alt.Scale(zero=False)),
        color=alt.Color('profit_percentage:Q', title='% Ganancia', scale=alt.Scale(type='log', domainMid=0, range='diverging')),
        tooltip=['title', 'movie_popularity', 'score', alt.Tooltip('profit_percentage', format='.1f')]
    ).properties(title='Popularidad (Log) vs. Puntaje, coloreado por % de Ganancia').interactive()
    st.altair_chart(scatter_pop_score, use_container_width=True)
    st.markdown("**Hallazgos Clave (Gráfico 2):** ... (pega tus hallazgos aquí) ...")
    st.divider()

    # Gráfico 3
    st.subheader("Gráfico 3: Distribución de Ingresos")
    histogram_revenue = alt.Chart(df_filtered).mark_bar().encode(
        x=alt.X('revenue:Q', bin=alt.Bin(maxbins=40), title='Ingresos ($)'),
        y=alt.Y('count()', title='Cantidad de Películas'),
        tooltip=[alt.Tooltip('revenue:Q', bin=True, title='Rango de Ingresos'), 'count()']
    ).properties(title='Distribución de Ingresos en Taquilla').interactive()
    st.altair_chart(histogram_revenue, use_container_width=True)
    st.markdown("**HallazGOS Clave (Gráfico 3):** ... (pega tus hallazgos aquí) ...")
    # --- FIN GRÁFICOS ---


# --- Pestaña 2: Prueba del Modelo (AHORA CON RESULTADOS) ---
with tab2:
    st.header("Prueba del Modelo en Vivo")
    st.write("Usa el formulario en la **barra lateral izquierda** para ingresar los datos de una película.")
    
    # --- MEJORA: Los resultados ahora se muestran aquí ---
    if st.session_state.prediction_made:
        st.subheader("Resultados de la Predicción")
        col1, col2 = st.columns(2)
        col1.metric(
            label="Ingreso (Revenue) Predicho", 
            value=f"${st.session_state.predicted_revenue:,.0f}"
        )
        col2.metric(
            label="Ganancia/Pérdida Estimada",
            value=f"${st.session_state.profit:,.0f}",
            delta=f"{st.session_state.profit_percent:.2f} %"
        )
        st.info(f"Cálculo basado en un presupuesto de ${st.session_state.budget:,.0f}.", icon="💰")
        
        st.divider()

        # --- ¡LA MAGIA! Gráfico SHAP Waterfall ---
        st.subheader("Explicación de la Predicción (XAI con SHAP)")
        st.write("Este gráfico muestra *por qué* el modelo llegó a esa predicción. Las características en **rojo** empujan la predicción hacia arriba (más ingresos), y las en **azul** la empujan hacia abajo.")
        
        # 1. Aplicar los mismos pasos del pipeline (imputación, escalado)
        input_transformed = model.named_steps['preprocessor'].transform(st.session_state.input_df)
        
        # 2. Obtener los valores SHAP del explainer
        # (El explainer fue creado sobre el regresor, así que usamos el input transformado)
        shap_values = shap_explainer.shap_values(input_transformed)
        
        # 3. Crear el gráfico
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0], # Valores SHAP para la primera (y única) predicción
                base_values=shap_explainer.expected_value, # El ingreso promedio del modelo
                data=st.session_state.input_df.iloc[0], # Los valores que ingresó el usuario
                feature_names=st.session_state.input_df.columns.tolist() # Nombres de las features
            ),
            max_display=9, # Mostrar las 9 features
            show=False # Evitar que se muestre con plt.show()
        )
        plt.tight_layout() # Ajustar el layout
        st.pyplot(fig) # Mostrar el gráfico en Streamlit
        
        with st.expander("Ver valores de entrada y SHAP"):
            st.write("Valores de entrada:")
            st.dataframe(st.session_state.input_df)
            st.write("Valores SHAP (la 'fuerza' de cada feature):")
            shap_df = pd.DataFrame(shap_values, columns=st.session_state.input_df.columns)
            st.dataframe(shap_df)

    else:
        st.info("Presiona el botón 'Predecir Ingresos' en la barra lateral para ver un resultado.")


# --- Pestaña 3: Sobre el Proyecto ---
with tab3:
    st.header("Detalles del Trabajo Práctico Integrador")
    # ... (pega aquí el contenido de tu pestaña "Sobre el Proyecto" anterior) ...
    st.subheader("Grupo 21")
    st.markdown("""
    * **Integrantes:**
    * *<Nombre Apellido 1>*
    * *<Nombre Apellido 2>*
    """)
    st.subheader("Contexto del Proyecto")
    st.markdown("...")