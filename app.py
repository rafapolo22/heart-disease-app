import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ====================== CONFIGURACIÓN DE PÁGINA ======================
st.set_page_config(
    page_title="CardioPredict | Predicción de Enfermedad Cardíaca",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS (Profesional) ======================
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subheader {
        color: #334155;
        font-weight: 500;
    }
    .card {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .result-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin: 15px 0;
    }
    .high-risk {
        background-color: #fee2e2;
        border: 2px solid #ef4444;
        color: #991b1b;
    }
    .low-risk {
        background-color: #ecfdf5;
        border: 2px solid #10b981;
        color: #065f46;
    }
    .stButton>button {
        width: 100%;
        height: 3.2rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ====================== CARGAR MODELOS ======================
@st.cache_resource
def load_models():
    log_model = joblib.load('logistic_heart.pkl')
    nn_model = joblib.load('nn_heart.pkl')
    scaler = joblib.load('scaler_heart.pkl')
    return log_model, nn_model, scaler

log_model, nn_model, scaler = load_models()

# ====================== SIDEBAR ======================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-with-pulse.png", width=80)  # Puedes cambiar por tu logo
    st.title("🫀 CardioPredict")
    st.markdown("### Herramienta de predicción de riesgo cardíaco")
    
    modelo_seleccionado = st.selectbox(
        "🤖 Modelo a utilizar:",
        ["Regresión Logística", "Red Neuronal (MLP)", "Ambos modelos"]
    )
    
    option = st.radio("Tipo de predicción:", 
                      ["1. Predicción Individual", "2. Predicción por Lotes"])

# ====================== TÍTULO PRINCIPAL ======================
st.markdown('<h1 class="main-header">Predicción de Enfermedad Cardíaca</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#64748b; font-size:1.1rem;">Usando Machine Learning para apoyar el diagnóstico clínico</p>', unsafe_allow_html=True)

# ====================== PREDICCIÓN INDIVIDUAL ======================
if option == "1. Predicción Individual":
    st.markdown("---")
    st.subheader("📋 Datos del Paciente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Edad", min_value=20, max_value=80, value=50)
        sex = st.selectbox("Sexo", [0, 1], format_func=lambda x: "👩 Mujer" if x == 0 else "👨 Hombre")
        cp = st.selectbox("Tipo de dolor de pecho (cp)", [0,1,2,3], 
                         format_func=lambda x: ["Típico", "Atípico", "No anginoso", "Asintomático"][x])
        trestbps = st.number_input("Presión arterial en reposo (mm Hg)", 80, 250, 120)
        chol = st.number_input("Colesterol sérico (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Azúcar en sangre > 120 mg/dl", [0,1], format_func=lambda x: "No" if x == 0 else "Sí")
    
    with col2:
        restecg = st.selectbox("Electrocardiograma en reposo", [0,1,2])
        thalach = st.number_input("Frecuencia cardíaca máxima", 60, 250, 150)
        exang = st.selectbox("Angina inducida por ejercicio", [0,1], format_func=lambda x: "No" if x == 0 else "Sí")
        oldpeak = st.number_input("Depresión del ST inducida por ejercicio", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Pendiente del segmento ST", [0,1,2])
        ca = st.selectbox("Número de vasos principales", [0,1,2,3])
        thal = st.selectbox("Thalassemia", [0,1,2,3])

    if st.button("🔮 Realizar Predicción", type="primary", use_container_width=True):
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                    thalach, exang, oldpeak, slope, ca, thal]],
                                  columns=['age','sex','cp','trestbps','chol','fbs','restecg',
                                           'thalach','exang','oldpeak','slope','ca','thal'])
        
        input_scaled = scaler.transform(input_data)

        if modelo_seleccionado == "Regresión Logística":
            pred = log_model.predict(input_scaled)[0]
            prob = log_model.predict_proba(input_scaled)[0][1]
            riesgo = "Alto" if pred == 1 else "Bajo"
            color_class = "high-risk" if pred == 1 else "low-risk"
            
            st.markdown(f"""
            <div class="result-box {color_class}">
                <h2>Resultado - Regresión Logística</h2>
                <h3>Riesgo: <strong>{riesgo}</strong></h3>
                <p>Probabilidad de enfermedad cardíaca: <strong>{prob:.1%}</strong></p>
            </div>
            """, unsafe_allow_html=True)

        # ... (puedes replicar el bloque para Red Neuronal y Ambos de forma similar)

else:
    # ====================== PREDICCIÓN POR LOTES ======================
    st.subheader("📤 Predicción por Lotes (Batch)")
    st.markdown("Sube un archivo **CSV** con las mismas columnas del dataset.")
    
    uploaded_file = st.file_uploader("Selecciona tu archivo CSV", type=["csv"])
    
    if uploaded_file:
        df_test = pd.read_csv(uploaded_file)
        st.success(f"✅ Archivo cargado correctamente: {len(df_test)} registros")
        st.dataframe(df_test.head(), use_container_width=True)

        # ... (mantén la lógica de predicción por lotes que ya tenías, solo mejora la presentación de resultados)
