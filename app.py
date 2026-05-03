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
        st.markdown("**👤 Datos Personales**")
        age = st.number_input("Edad", 20, 80, 50)
        sex = st.selectbox("Sexo", [0,1], format_func=lambda x: "Mujer" if x==0 else "Hombre")
        cp = st.selectbox("Dolor de pecho (cp)", [0,1,2,3],
            format_func=lambda x: {0:"Angina típica", 1:"Angina atípica", 2:"No anginoso", 3:"Asintomático"}[x])
        fbs = st.selectbox("Azúcar en sangre >120", [0,1], format_func=lambda x: "No" if x==0 else "Sí")
 
    with col2:
        st.markdown("**🩺 Datos Clínicos**")
        trestbps = st.number_input("Presión arterial (trestbps)", 80, 250, 120)
        chol = st.number_input("Colesterol (chol)", 100, 600, 200)
        restecg = st.selectbox("Electrocardiograma (restecg)", [0,1,2],
            format_func=lambda x: {0:"Normal", 1:"Anomalía ST-T", 2:"Hipertrofia"}[x])
        exang = st.selectbox("Angina por ejercicio", [0,1], format_func=lambda x: "No" if x==0 else "Sí")
 
    with col3:
        st.markdown("**❤️ Datos Cardíacos**")
        thalach = st.number_input("Frecuencia cardíaca máx.", 60, 250, 150)
        oldpeak = st.number_input("Depresión ST (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Pendiente ST (slope)", [0,1,2],
            format_func=lambda x: {0:"Ascendente", 1:"Plana", 2:"Descendente"}[x])
        ca = st.selectbox("Vasos principales (ca)", [0,1,2,3])
        thal = st.selectbox("Thalassemia (thal)", [0,1,2,3],
            format_func=lambda x: {0:"Normal", 1:"Defecto fijo", 2:"Reversible", 3:"Sin info"}[x])
 
    st.markdown("---")
 
    col_btn = st.columns([2,1,2])
    with col_btn[1]:
        analizar = st.button("🔮 Analizar Riesgo")
 
    if analizar:
        errores = []
        if trestbps < 80 or trestbps > 250:
            errores.append("⚠️ Presión arterial debe estar entre 80 y 250.")
        if chol < 100 or chol > 600:
            errores.append("⚠️ Colesterol debe estar entre 100 y 600.")
        if thalach < 60 or thalach > 250:
            errores.append("⚠️ Frecuencia cardíaca debe estar entre 60 y 250.")
        if oldpeak < 0.0 or oldpeak > 10.0:
            errores.append("⚠️ Depresión del ST debe estar entre 0.0 y 10.0.")
 
        if errores:
            for e in errores:
                st.error(e)
        else:
            with st.spinner("Analizando datos..."):
                input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                            thalach, exang, oldpeak, slope, ca, thal]],
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
                    prob_log = log_model.predict_proba(input_scaled)[0][1]
                    prob_nn = nn_model.predict_proba(input_scaled)[0][1]
                    prob = (prob_log + prob_nn) / 2
                    pred = int(prob > 0.5)
 
            st.subheader("📊 Resultado")
            col1, col2 = st.columns(2)
 
            with col1:
                st.metric("Probabilidad de Enfermedad", f"{prob:.2%}")
 
            with col2:
                if pred == 1:
                    st.error("⚠️ Alto Riesgo Cardíaco")
                else:
                    st.success("✅ Bajo Riesgo Cardíaco")
 
            st.progress(int(prob * 100))
 
# ================= LOTES =================
else:
    st.subheader("📂 Carga de Datos por Lotes")
    st.info("📌 El CSV debe tener las columnas del dataset Cleveland. Puede incluir 'target' para ver métricas.")
 
    uploaded_file = st.file_uploader("Sube tu CSV (hasta 200MB)", type=["csv"])
 
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown("#### 👀 Vista previa")
        st.dataframe(df.head(), use_container_width=True)
        st.caption(f"Total de registros: {len(df)}")
 
        X = df.drop(columns=['target'], errors='ignore')
 
        for col in ['ca', 'thal']:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].fillna(X[col].mean())
 
        X_scaled = scaler.transform(X)
 
        modelos = []
        if modelo_seleccionado in ["Regresión Logística", "Ambos"]:
            modelos.append(("Regresión Logística", log_model, "Blues"))
        if modelo_seleccionado in ["Red Neuronal (MLP)", "Ambos"]:
            modelos.append(("Red Neuronal (MLP)", nn_model, "Oranges"))
 
        for nombre, modelo, cmap in modelos:
            st.markdown("---")
            st.markdown(f"#### 📊 Resultados - {nombre}")
            pred = modelo.predict(X_scaled)
 
            r1, r2 = st.columns(2)
            with r1:
                conteo = pd.Series(pred).value_counts().rename({0: "Sin enfermedad", 1: "Con enfermedad"})
                st.bar_chart(conteo)
            with r2:
                if 'target' in df.columns:
                    st.markdown("**Reporte de Clasificación:**")
                    st.text(classification_report(df['target'], pred))
 
            if 'target' in df.columns:
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.heatmap(confusion_matrix(df['target'], pred),
                            annot=True, fmt='d', cmap=cmap, ax=ax, linewidths=0.5)
                ax.set_title(f"Matriz de Confusión - {nombre}", fontsize=11)
                ax.set_xlabel("Predicho")
                ax.set_ylabel("Real")
                st.pyplot(fig)
