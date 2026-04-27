import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ====================== Cargar modelos ======================
@st.cache_resource
def load_models():
    log_model = joblib.load('logistic_heart.pkl')
    nn_model = joblib.load('nn_heart.pkl')
    scaler = joblib.load('scaler_heart.pkl')
    return log_model, nn_model, scaler

log_model, nn_model, scaler = load_models()

# ====================== Título ======================
st.title("🫀 Predicción de Enfermedad Cardíaca - Proyecto Final")
st.markdown("**Modelos:** Regresión Logística + Red Neuronal (MLP)")

option = st.radio("Elige el tipo de predicción:", 
                  ["1. Predicción Individual", "2. Predicción por Lotes"])

# ====================== PREDICCIÓN INDIVIDUAL ======================
if option == "1. Predicción Individual":
    st.header("Ingresa los datos del paciente")

    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Edad", min_value=20, max_value=80, value=50)
        sex = st.selectbox("Sexo", [0, 1], format_func=lambda x: "Mujer" if x == 0 else "Hombre")
        cp = st.selectbox("Tipo de dolor de pecho (cp)", [0,1,2,3])
        trestbps = st.number_input("Presión arterial en reposo (trestbps)", value=120)
        chol = st.number_input("Colesterol (chol)", value=200)
        fbs = st.selectbox("Azúcar en sangre >120 (fbs)", [0,1])
    
    with col2:
        restecg = st.selectbox("Electrocardiograma en reposo (restecg)", [0,1,2])
        thalach = st.number_input("Frecuencia cardíaca máxima (thalach)", value=150)
        exang = st.selectbox("Angina inducida por ejercicio (exang)", [0,1])
        oldpeak = st.number_input("Depresión del ST (oldpeak)", value=1.0, step=0.1)
        slope = st.selectbox("Pendiente del ST (slope)", [0,1,2])
        ca = st.selectbox("Vasos principales (ca)", [0,1,2,3])
        thal = st.selectbox("Thalassemia (thal)", [0,1,2,3])  # Nota: en algunos datasets es 3,6,7 → ajusta si es necesario

    if st.button("🔮 Predecir", type="primary"):
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, 
                                    thalach, exang, oldpeak, slope, ca, thal]],
                                  columns=['age','sex','cp','trestbps','chol','fbs','restecg',
                                           'thalach','exang','oldpeak','slope','ca','thal'])
        
        input_scaled = scaler.transform(input_data)
        
        pred_log = log_model.predict(input_scaled)[0]
        pred_nn = nn_model.predict(input_scaled)[0]
        
        prob_log = log_model.predict_proba(input_scaled)[0][1]
        prob_nn = nn_model.predict_proba(input_scaled)[0][1]
        
        st.success(f"**Regresión Logística:** {'Tiene enfermedad' if pred_log == 1 else 'No tiene enfermedad'} ({prob_log:.1%} probabilidad)")
        st.success(f"**Red Neuronal:** {'Tiene enfermedad' if pred_nn == 1 else 'No tiene enfermedad'} ({prob_nn:.1%} probabilidad)")

# ====================== PREDICCIÓN POR LOTES ======================
else:
    st.header("Sube un archivo CSV para predicción por lotes")
    st.markdown("El CSV debe tener las mismas columnas que el dataset (sin la columna 'target')")
    
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    
    if uploaded_file is not None:
        df_test = pd.read_csv(uploaded_file)
        st.write("Vista previa de los datos:", df_test.head())
        
        X_test_scaled = scaler.transform(df_test)
        
        pred_log = log_model.predict(X_test_scaled)
        pred_nn = nn_model.predict(X_test_scaled)
        
        # Mostrar resultados
        st.subheader("Resultados Regresión Logística")
        st.write(pd.Series(pred_log, name="Predicción_Log").value_counts())
        
        st.subheader("Resultados Red Neuronal")
        st.write(pd.Series(pred_nn, name="Predicción_NN").value_counts())
        
        # Si tienes las etiquetas reales (columna 'target'), muestra métricas
        if 'target' in df_test.columns:
            y_true = df_test['target']
            st.subheader("Métricas - Regresión Logística")
            st.text(classification_report(y_true, pred_log))
            
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_true, pred_log), annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title("Matriz de Confusión - Regresión Logística")
            st.pyplot(fig)
            
            st.subheader("Métricas - Red Neuronal")
            st.text(classification_report(y_true, pred_nn))
            
            fig2, ax2 = plt.subplots()
            sns.heatmap(confusion_matrix(y_true, pred_nn), annot=True, fmt='d', cmap='Oranges', ax=ax2)
            ax2.set_title("Matriz de Confusión - Red Neuronal")
            st.pyplot(fig2)

