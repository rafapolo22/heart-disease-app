import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ================= CONFIGURACIÓN =================
st.set_page_config(
    page_title="Predicción Cardíaca",
    page_icon="🫀",
    layout="wide"
)

# ================= ESTILO PERSONALIZADO =================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.stButton>button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ================= CARGAR MODELOS =================
@st.cache_resource
def load_models():
    log_model = joblib.load('logistic_heart.pkl')
    nn_model = joblib.load('nn_heart.pkl')
    scaler = joblib.load('scaler_heart.pkl')
    return log_model, nn_model, scaler

log_model, nn_model, scaler = load_models()

# ================= SIDEBAR =================
st.sidebar.title("⚙️ Configuración")
modelo_seleccionado = st.sidebar.selectbox(
    "Selecciona modelo",
    ["Regresión Logística", "Red Neuronal (MLP)", "Ambos"]
)

option = st.sidebar.radio(
    "Modo de uso",
    ["Predicción Individual", "Predicción por Lotes"]
)

# ================= HEADER =================
st.title("🫀 Sistema Inteligente de Predicción Cardíaca")
st.markdown("### Análisis basado en Machine Learning")

# ================= INDIVIDUAL =================
if option == "Predicción Individual":

    st.subheader("📋 Datos del Paciente")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Edad", 20, 80, 50)
        sex = st.selectbox("Sexo", [0,1], format_func=lambda x: "Mujer" if x==0 else "Hombre")
        cp = st.selectbox("Dolor de pecho", [0,1,2,3])

    with col2:
        trestbps = st.number_input("Presión arterial", 80, 250, 120)
        chol = st.number_input("Colesterol", 100, 600, 200)
        fbs = st.selectbox("Azúcar alta", [0,1])

    with col3:
        thalach = st.number_input("Frecuencia cardíaca", 60, 250, 150)
        exang = st.selectbox("Angina", [0,1])
        oldpeak = st.number_input("ST depresión", 0.0, 10.0, 1.0)

    # Botón
    if st.button("🔮 Analizar Riesgo"):

        with st.spinner("Analizando datos..."):
            input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, 0,
                                        thalach, exang, oldpeak, 0, 0, 1]],
                                      columns=['age','sex','cp','trestbps','chol','fbs','restecg',
                                               'thalach','exang','oldpeak','slope','ca','thal'])

            input_scaled = scaler.transform(input_data)

            if modelo_seleccionado == "Regresión Logística":
                pred = log_model.predict(input_scaled)[0]
                prob = log_model.predict_proba(input_scaled)[0][1]

            elif modelo_seleccionado == "Red Neuronal (MLP)":
                pred = nn_model.predict(input_scaled)[0]
                prob = nn_model.predict_proba(input_scaled)[0][1]

            else:
                prob = (log_model.predict_proba(input_scaled)[0][1] +
                        nn_model.predict_proba(input_scaled)[0][1]) / 2
                pred = int(prob > 0.5)

        # ================= RESULTADO VISUAL =================
        st.subheader("📊 Resultado")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Probabilidad de Enfermedad", f"{prob:.2%}")

        with col2:
            if pred == 1:
                st.error("⚠️ Alto Riesgo Cardíaco")
            else:
                st.success("✅ Bajo Riesgo")

        # Barra de progreso
        st.progress(int(prob * 100))

# ================= LOTES =================
else:

    st.subheader("📂 Carga de Datos")

    uploaded_file = st.file_uploader("Sube tu CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(), use_container_width=True)

        X = df.drop(columns=['target'], errors='ignore')
        X_scaled = scaler.transform(X)

        pred = log_model.predict(X_scaled)

        st.subheader("📊 Resultados")
        st.write(pd.Series(pred).value_counts())

        if 'target' in df.columns:
            st.subheader("📉 Evaluación del Modelo")

            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(df['target'], pred),
                        annot=True, fmt='d', ax=ax)
            st.pyplot(fig)
