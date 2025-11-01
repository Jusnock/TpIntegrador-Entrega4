import streamlit as st
import pandas as pd
import joblib
import altair as alt
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import requests  # Necesario para la descarga

# --- 1. Configuración de la Descarga ---
# ID de tu archivo de Google Drive (Corregido)
GOOGLE_DRIVE_FILE_ID = "1C33INLaAQi3cJSpwWcnaOc0dDwHEJ2CC"

MODEL_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
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
    layout="wide",
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
    """Carga el dataset de películas."""
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
    
@st.cache_resource
def get_shap_explainer(_model):
    """Crea y cachea el explainer de SHAP para el modelo de árbol."""
    try:
        regressor = _model.named_steps['regressor']
        return shap.TreeExplainer(regressor)
    except Exception as e:
        st.error(f"Error al crear el SHAP Explainer: {e}")
        st.info("Asegúrate de que el modelo cargado sea un pipeline con un 'regressor' (CatBoost).")
        st.stop()

# --- 5. INICIO DE LA APP ---

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

# --- CAMBIO: Diccionario para nombres amigables ---
feature_name_map = {
    'score': 'Puntaje TMDB',
    'movie_popularity': 'Popularidad (Película)',
    'actor1_popularity': 'Popularidad (Actor 1)',
    'actor2_popularity': 'Popularidad (Actor 2)',
    'actor3_popularity': 'Popularidad (Actor 3)',
    'budget': 'Presupuesto',
    'actor1_age': 'Edad (Actor 1)',
    'actor2_age': 'Edad (Actor 2)',
    'actor3_age': 'Edad (Actor 3)'
}


# --- Título Principal y Pestañas (Tabs) ---
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
    
    st.subheader("Filtros del Dataset")
    df_sample = df_raw.sample(n=2000, random_state=42).copy()
    budget_range = st.slider(
        "Filtrar por Presupuesto (Budget)",
        min_value=float(df_sample['budget'].min()),
        max_value=float(df_sample['budget'].max()),
        value=(float(df_sample['budget'].min()), float(df_sample['budget'].max())),
        key="budget_slider_tab1"
    )
    df_filtered = df_sample[
        (df_sample['budget'] >= budget_range[0]) & 
        (df_sample['budget'] <= budget_range[1]) &
        (df_sample['revenue'] > 0) &
        (df_sample['movie_popularity'] > 0) 
    ]
    st.write(f"Mostrando {len(df_filtered)} películas (de 2000 aleatorias).")

    # --- Gráfico 1 ---
    st.subheader("Gráfico 1: Presupuesto vs. Ingresos (con Línea de Rentabilidad)")
    line_data = pd.DataFrame({'budget': [df_filtered['budget'].min(), df_filtered['budget'].max()], 'revenue': [df_filtered['budget'].min(), df_filtered['budget'].max()]})
    profit_line = alt.Chart(line_data).mark_line(color='red', strokeDash=[5,5]).encode(x='budget:Q', y='revenue:Q')
    
    scatter_budget_revenue = alt.Chart(df_filtered).mark_circle(opacity=0.6).encode(
        x=alt.X('budget:Q', title='Presupuesto ($)', axis=alt.Axis(format='$,.0f')),
        y=alt.Y('revenue:Q', title='Ingresos ($)', axis=alt.Axis(format='$,.0f')),
        color=alt.Color('score:Q', title='Puntaje TMDB', scale=alt.Scale(range='heatmap')),
        # --- CAMBIO: Añadir colores a la leyenda de TAMAÑO para que sea visible ---
        size=alt.Size('profit_percentage:Q', title='% Ganancia', 
                      legend=alt.Legend(format='.0f', 
                                        symbolFillColor='lightgray', # Color de relleno del círculo
                                        symbolStrokeColor='white', # Color del borde del círculo
                                        labelColor='white',        # Color del texto (0, 1000, etc.)
                                        titleColor='white'         # Color del título (% Ganancia)
                                       )
                     ),
        tooltip=['title', alt.Tooltip('budget', format='$,.0f'), alt.Tooltip('revenue', format='$,.0f'), 'score', alt.Tooltip('profit_percentage', format='.1f')]
    ).interactive()
    
    final_chart_1 = scatter_budget_revenue + profit_line
    st.altair_chart(final_chart_1, use_container_width=True)
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
                        scale=alt.Scale(domainMid=0, range='diverging')), 
        tooltip=['title', 'movie_popularity', 'score', alt.Tooltip('profit_percentage', format='.1f')]
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
        tooltip=[alt.Tooltip('revenue:Q', bin=True, title='Rango de Ingresos'), 'count()']
    ).properties(
        title='Distribución de Ingresos en Taquilla'
    ).interactive()
    st.altair_chart(histogram_revenue, use_container_width=True)
    st.markdown("""
    **Hallazgos Clave (Gráfico 3):**
    * La gran mayoría de las películas de acción recaudan menos de $250 millones.
    * El dataset tiene una fuerte "cola larga" (long tail), con unas pocas películas (los "blockbusters") que recaudan cantidades extremadamente altas (ej. +$750M).
    """)


# --- Pestaña 2: Prueba del Modelo (AHORA CON FORMULARIO) ---
with tab2:
    st.header("Prueba del Modelo en Vivo")
    st.write("Ingresa los datos de una película en el formulario para obtener una predicción de sus ingresos y la explicación del modelo.")
    
    col1, col2 = st.columns([1, 2]) # Columna de inputs más pequeña

    with col1:
        with st.form(key="prediction_form_tab2"):
            st.header("Datos de la Película")
            budget = st.number_input("Presupuesto (Budget)", min_value=1000000, max_value=400000000, value=50000000, step=1000000, format="%d")
            score = st.slider("Puntaje TMDB (Score)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
            # --- Corregido el rango de movie_popularity basado en tu código ---
            movie_popularity = st.slider("Popularidad de la Película", min_value=1.0, max_value=500.0, value=100.0, step=5.0) 
            
            st.header("Datos de Actores")
            actor1_popularity = st.slider("Popularidad Actor 1", min_value=1.0, max_value=100.0, value=15.0)
            actor1_age = st.slider("Edad Actor 1", min_value=18, max_value=80, value=45)
            
            # --- Corregido el rango de actor2_popularity basado en tu código ---
            actor2_popularity = st.slider("Popularidad Actor 2", min_value=1.0, max_value=100.0, value=10.0) 
            actor2_age = st.slider("Edad Actor 2", min_value=18, max_value=80, value=40)

            # --- Corregido el rango de actor3_popularity basado en tu código ---
            actor3_popularity = st.slider("Popularidad Actor 3", min_value=1.0, max_value=100.0, value=5.0) 
            actor3_age = st.slider("Edad Actor 3", min_value=18, max_value=80, value=35)

            submit_button_tab2 = st.form_submit_button(label="📈 Predecir Ingresos", type="primary", use_container_width=True)

    with col2:
        if submit_button_tab2:
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

            # --- Mostrar Resultados (con formato en Millones) ---
            st.subheader("Resultados de la Predicción")
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric(
                label="Ingreso (Revenue) Predicho", 
                value=f"${predicted_revenue / 1_000_000:,.1f} Millones"
            )
            metric_col2.metric(
                label="Ganancia/Pérdida Estimada",
                value=f"${profit / 1_000_000:,.1f} Millones",
                delta=f"{profit_percent:.2f} %"
            )
            st.info(f"Cálculo basado en un presupuesto de ${budget:,.0f}.", icon="💰")
            
            st.divider()

            # --- Nuevo Gráfico SHAP con Altair (más legible) ---
            st.subheader("Explicación de la Predicción (XAI)")
            st.write("Este gráfico muestra cuánto *impactó* cada característica en la predicción final. Las barras **rojas** sumaron ingresos, las **azules** restaron.")

            try:
                # Calcular valores SHAP
                input_transformed = model.named_steps['preprocessor'].transform(input_df)
                shap_values = shap_explainer.shap_values(input_transformed)
                base_value = shap_explainer.expected_value
                shap_values_list = shap_values[0]
                feature_names = input_df.columns.tolist()
                feature_values = input_df.iloc[0].values

                # Formatear valores de features para las etiquetas
                feature_values_str = []
                for name, val in zip(feature_names, feature_values):
                    if name in ['score']:
                        feature_values_str.append(f"{val:.1f}")
                    elif name in ['movie_popularity', 'actor1_popularity', 'actor2_popularity', 'actor3_popularity']:
                        feature_values_str.append(f"{val:.2f}")
                    else: # budget, ages
                        feature_values_str.append(f"{val:,.0f}")
                
                # --- CAMBIO: Usar el feature_name_map para etiquetas amigables ---
                feature_labels = [f"{feature_name_map.get(name, name)} = {val}" for name, val in zip(feature_names, feature_values_str)]
                
                # Crear DataFrame para el gráfico
                shap_df = pd.DataFrame({
                    'feature_label': feature_labels,
                    'Impacto en Ingresos ($)': shap_values_list,
                    'feature_name': feature_names # Guardamos el nombre original para formateo
                })
                
                # Añadir color para el gráfico
                shap_df['Impacto'] = np.where(shap_df['Impacto en Ingresos ($)'] >= 0, 'Positivo', 'Negativo')
                
                # Crear el gráfico de barras horizontal con Altair
                chart = alt.Chart(shap_df).mark_bar().encode(
                    # Ordenar las barras por el valor absoluto del impacto, de mayor a menor
                    x=alt.X('Impacto en Ingresos ($)', 
                            axis=alt.Axis(format='$,.0f') # Formato de moneda
                           ),
                    y=alt.Y('feature_label', 
                            title="Característica y Valor Ingresado",
                            # Ordenar por valor absoluto descendente
                            sort=alt.EncodingSortField(field="Impacto en Ingresos ($)", op="sum", order='descending') 
                           ),
                    color=alt.Color('Impacto', 
                                    scale={
                                        'domain': ['Positivo', 'Negativo'],
                                        'range': ['#e45756', '#3b5b9a'] # Rojo, Azul
                                    },
                                    legend=alt.Legend(title="Tipo de Impacto")
                                   ),
                    tooltip=[
                        alt.Tooltip('feature_label', title='Característica'),
                        alt.Tooltip('Impacto en Ingresos ($)', title='Impacto', format='$,.0f')
                    ]
                ).properties(
                    title='Contribución de cada Característica a la Predicción'
                ).interactive()

                st.altair_chart(chart, use_container_width=True)

                # Mostrar el valor base y la predicción final
                st.info(f"**Predicción Base (Promedio del Modelo):** ${base_value:,.0f}\n"
                        f"**Predicción Final (Base + Impactos):** ${predicted_revenue:,.0f}", icon="💡")

                
                # --- Formatear las tablas en el expander ---
                with st.expander("Ver valores de entrada y SHAP (datos brutos)"):
                    st.write("Valores de entrada (los datos que ingresaste):")
                    # Formatear input_df para que se vea más limpio
                    st.dataframe(input_df.style.format({
                        'score': '{:.1f}',
                        'movie_popularity': '{:.2f}',
                        'actor1_popularity': '{:.2f}',
                        'actor2_popularity': '{:.2f}',
                        'actor3_popularity': '{:.2f}',
                        'budget': '${:,.0f}',
                        'actor1_age': '{:.0f}',
                        'actor2_age': '{:.0f}',
                        'actor3_age': '{:.0f}'
                    }))
                    
                    st.write("Valores SHAP (el impacto de cada feature, en $):")
                    st.markdown("*(Nota: Un valor negativo, como en `actor2_age`, no significa que la edad sea negativa, sino que esa edad **restó** ese monto a la predicción final.)*")
                    shap_df_raw = pd.DataFrame(shap_values, columns=input_df.columns)
                    # Formatear shap_df_raw para que muestre valores enteros de moneda
                    st.dataframe(shap_df_raw.style.format('${:,.0f}'))
            
            except Exception as e:
                st.error(f"Error al generar el gráfico SHAP: {e}")
                st.write("El modelo se cargó, pero no se pudo generar la explicación.")
                # Imprimir el error en la consola de Streamlit para depuración
                print(f"Error SHAP: {e}")

        else:
            st.info("Ingresa los datos en el formulario de la izquierda y presiona 'Predecir Ingresos' para ver los resultados.")


# --- Pestaña 3: Sobre el Proyecto ---
with tab3:
    st.header("Detalles del Trabajo Práctico Integrador")
    
    st.subheader("Grupo 21")
    st.markdown("""
    * **Integrantes:**
    * *Secotaro,Leonardo*
    * *Vazquez,Juan Francisco*
    * *Melonari,Martin*
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
    
    El modelo final (después de la optimización y el tratamiento de outliers) alcanzó un **$R^2 \\approx 0.79$** en el conjunto de prueba.
    """)
    
    st.subheader("Repositorio del Proyecto")
    st.markdown("El código fuente de esta aplicación y de los notebooks de análisis se encuentra en [GitHub](https://github.com/jusnock/tpintegrador-entrega4).")

