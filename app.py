import io
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from fpdf import FPDF
import seaborn as sns
import matplotlib.pyplot as plt
 
# ================= CONFIGURACIÓN =================
st.set_page_config(
    page_title='Predicción Cardíaca',
    page_icon='🫀',
    layout='wide'
)
 
# ================= ESTILO PERSONALIZADO =================
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
.stButton>button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
@media (max-width: 768px) {
    .css-1d391kg {
        padding: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)
 
FEATURE_COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                   'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
 
CATEGORY_MAPS = {
    'sex': {
        'm': 1, 'male': 1, 'hombre': 1, 'masculino': 1,
        'f': 0, 'female': 0, 'mujer': 0, 'femenino': 0
    },
    'cp': {
        'angina típica': 0, 'angina atípica': 1, 'no anginoso': 2, 'asintomático': 3,
        'angina tipica': 0, 'angina atipica': 1
    },
    'restecg': {
        'normal': 0, 'anomalía st-t': 1, 'anomalía st t': 1, 'hipertrofia': 2
    },
    'exang': {
        'no': 0, 'sí': 1, 'si': 1, 'yes': 1
    },
    'slope': {
        'ascendente': 0, 'plana': 1, 'descendente': 2
    },
    'thal': {
        'normal': 0, 'defecto fijo': 1, 'reversible': 2, 'sin info': 3, 'sin información': 3
    },
    'fbs': {
        'no': 0, 'sí': 1, 'si': 1, 'yes': 1
    }
}
 
# ================= CARGAR MODELOS =================
@st.cache_resource
def load_models():
    try:
        log_model = joblib.load('logistic_heart.pkl')
        nn_model = joblib.load('nn_heart.pkl')
        scaler = joblib.load('scaler_heart.pkl')
    except FileNotFoundError as err:
        st.error(f"No se encuentran los archivos de modelo: {err}")
        st.stop()
    return log_model, nn_model, scaler
 
log_model, nn_model, scaler = load_models()
 
# ================= UTILIDADES =================

def clean_batch_data(df: pd.DataFrame):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    df.replace('?', np.nan, inplace=True)
 
    for col, mapping in CATEGORY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace(mapping)
 
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')
 
    if 'target' in df.columns:
        df['target'] = pd.to_numeric(df['target'], errors='coerce')
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
 
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    constant_cols = [c for c in numeric_cols if df[c].nunique(dropna=False) <= 1]
    df.drop(columns=constant_cols, inplace=True, errors='ignore')
 
    if {'age', 'chol'}.issubset(df.columns):
        df['chol_age_ratio'] = df['chol'] / df['age']
    if {'trestbps', 'thalach'}.issubset(df.columns):
        df['pulse_diff'] = (df['trestbps'] - df['thalach']).abs()
 
    df[numeric_cols] = df[numeric_cols].astype(float)
    df = df.fillna(df.mean(numeric_only=True))
    return df, constant_cols
 

def build_prediction_report(input_data: pd.DataFrame, pred: int, prob: float, model_name: str):
    report = input_data.copy()
    report['modelo'] = model_name
    report['probabilidad'] = prob
    report['prediccion'] = pred
    report['interpretacion'] = report['prediccion'].apply(lambda x: 'Bajo Riesgo' if x == 0 else 'Alto Riesgo')
    return report
 

def build_pdf_report(input_data: pd.DataFrame, pred: int, prob: float, model_name: str):
    row = input_data.iloc[0].to_dict()
    interpretacion = 'Bajo Riesgo' if pred == 0 else 'Alto Riesgo'
    pdf = FPDF(format='letter')
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'Reporte de Predicción Cardíaca', ln=True, align='C')
    pdf.ln(5)
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 8, f'Modelo: {model_name}', ln=True)
    pdf.cell(0, 8, f'Probabilidad: {prob:.2%}', ln=True)
    pdf.cell(0, 8, f'Predicción: {pred}', ln=True)
    pdf.cell(0, 8, f'Interpretación: {interpretacion}', ln=True)
    pdf.ln(5)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Valores de entrada:', ln=True)
    pdf.set_font('Helvetica', '', 11)
    for key, value in row.items():
        pdf.multi_cell(0, 6, f'- {key}: {value}')
    return pdf.output(dest='S').encode('latin-1')
 

def build_batch_pdf_report(output: pd.DataFrame, model_name: str, target_present: bool):
    pdf = FPDF(format='letter')
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'Reporte de Resultados por Lote', ln=True, align='C')
    pdf.ln(5)
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 8, f'Modelo: {model_name}', ln=True)
    pdf.cell(0, 8, f'Registros procesados: {len(output)}', ln=True)
    if target_present:
        report_col = [c for c in output.columns if c.startswith('pred_')]
        prob_col = [c for c in output.columns if c.startswith('prob_')]
        if report_col:
            pdf.ln(3)
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 8, 'Métricas de Evaluación:', ln=True)
            pdf.set_font('Helvetica', '', 11)
            report = classification_report(output['target'], output[report_col[-1]],
                                           target_names=['Sin Enfermedad (0)', 'Con Enfermedad (1)'])
            for line in report.split('\n'):
                pdf.multi_cell(0, 6, line)
    else:
        pdf.ln(3)
        pdf.cell(0, 8, 'No se proporcionó columna target para métricas.', ln=True)
    pdf.ln(3)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Primeros 10 registros con predicción:', ln=True)
    pdf.set_font('Helvetica', '', 10)
    sample = output.head(10).to_string(index=False)
    for line in sample.split('\n'):
        pdf.multi_cell(0, 5, line)
    return pdf.output(dest='S').encode('latin-1')
 

def get_prediction_results(X_scaled: np.ndarray, modelo_seleccionado: str):
    if modelo_seleccionado == 'Regresión Logística':
        pred = log_model.predict(X_scaled)
        prob = log_model.predict_proba(X_scaled)[:, 1]
        return [('Regresión Logística', pred, prob)]
    if modelo_seleccionado == 'Red Neuronal (MLP)':
        pred = nn_model.predict(X_scaled)
        prob = nn_model.predict_proba(X_scaled)[:, 1]
        return [('Red Neuronal (MLP)', pred, prob)]
    prob_log = log_model.predict_proba(X_scaled)[:, 1]
    prob_nn = nn_model.predict_proba(X_scaled)[:, 1]
    combined_prob = (prob_log + prob_nn) / 2
    combined_pred = (combined_prob > 0.5).astype(int)
    return [
        ('Regresión Logística', log_model.predict(X_scaled), prob_log),
        ('Red Neuronal (MLP)', nn_model.predict(X_scaled), prob_nn),
        ('Ensamblado', combined_pred, combined_prob)
    ]
 
# ================= SIDEBAR =================
st.sidebar.title('⚙️ Configuración')
modelo_seleccionado = st.sidebar.selectbox(
    'Selecciona modelo',
    ['Regresión Logística', 'Red Neuronal (MLP)', 'Ambos']
)
option = st.sidebar.radio('Modo de uso', ['Predicción Individual', 'Predicción por Lotes'])
 
# ================= HEADER =================
st.title('🫀 Sistema Inteligente de Predicción Cardíaca')
st.markdown('### Análisis de datos con limpieza, modelos y reporte descargable')
 
if option == 'Predicción Individual':
    st.subheader('📋 Datos del Paciente')
    col1, col2, col3 = st.columns([1, 1, 1])
 
    with col1:
        age = st.number_input('Edad', 20, 80, 50, step=1)
        sex = st.selectbox('Sexo', [0, 1], format_func=lambda x: 'Mujer' if x == 0 else 'Hombre')
        cp = st.selectbox('Dolor de pecho (cp)', [0, 1, 2, 3],
                          format_func=lambda x: {0: 'Angina típica', 1: 'Angina atípica', 2: 'No anginoso', 3: 'Asintomático'}[x])
        fbs = st.selectbox('Azúcar en sangre >120', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Sí')
 
    with col2:
        trestbps = st.number_input('Presión arterial (trestbps)', 80, 250, 120, step=1)
        chol = st.number_input('Colesterol (chol)', 100, 600, 200, step=1)
        restecg = st.selectbox('Electrocardiograma (restecg)', [0, 1, 2],
                               format_func=lambda x: {0: 'Normal', 1: 'Anomalía ST-T', 2: 'Hipertrofia'}[x])
        exang = st.selectbox('Angina por ejercicio', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Sí')
 
    with col3:
        thalach = st.number_input('Frecuencia cardíaca máx.', 60, 250, 150, step=1)
        oldpeak = st.number_input('Depresión ST (oldpeak)', 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox('Pendiente ST (slope)', [0, 1, 2],
                             format_func=lambda x: {0: 'Ascendente', 1: 'Plana', 2: 'Descendente'}[x])
        ca = st.selectbox('Vasos principales (ca)', [0, 1, 2, 3])
        thal = st.selectbox('Thalassemia (thal)', [0, 1, 2, 3],
                            format_func=lambda x: {0: 'Normal', 1: 'Defecto fijo', 2: 'Reversible', 3: 'Sin info'}[x])
 
    st.markdown('---')
    analizar = st.button('🔮 Analizar Riesgo')
 
    if analizar:
        errores = []
        if not 20 <= age <= 80:
            errores.append('⚠️ Edad debe estar entre 20 y 80 años.')
        if not 80 <= trestbps <= 250:
            errores.append('⚠️ Presión arterial debe estar entre 80 y 250.')
        if not 100 <= chol <= 600:
            errores.append('⚠️ Colesterol debe estar entre 100 y 600.')
        if not 60 <= thalach <= 250:
            errores.append('⚠️ Frecuencia cardíaca debe estar entre 60 y 250.')
        if not 0.0 <= oldpeak <= 10.0:
            errores.append('⚠️ Depresión del ST debe estar entre 0.0 y 10.0.')
 
        if errores:
            for e in errores:
                st.error(e)
        else:
            input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                        thalach, exang, oldpeak, slope, ca, thal]],
                                      columns=FEATURE_COLUMNS)
            input_scaled = scaler.transform(input_data)
            results = get_prediction_results(input_scaled, modelo_seleccionado)
 
            for model_name, pred_array, prob_array in results:
                pred = int(pred_array[0])
                prob = float(prob_array[0])
                st.subheader(f'📊 Resultado - {model_name}')
                c1, c2 = st.columns(2)
                with c1:
                    st.metric('Probabilidad de Enfermedad', f'{prob:.2%}')
                with c2:
                    if pred == 1:
                        st.error('⚠️ Alto Riesgo Cardíaco — Clase 1: Con Enfermedad')
                    else:
                        st.success('✅ Bajo Riesgo Cardíaco — Clase 0: Sin Enfermedad')
                st.progress(int(prob * 100))
 
                report = build_prediction_report(input_data, pred, prob, model_name)
                csv = report.to_csv(index=False).encode('utf-8')
                pdf = build_pdf_report(input_data, pred, prob, model_name)
                st.download_button('📥 Descargar reporte CSV', csv,
                                   file_name=f'reporte_prediccion_{model_name.replace(" ", "_")}.csv',
                                   mime='text/csv')
                st.download_button('📄 Descargar reporte PDF', pdf,
                                   file_name=f'reporte_prediccion_{model_name.replace(" ", "_")}.pdf',
                                   mime='application/pdf')
 
            st.markdown('---')
            st.markdown('#### 🏷️ Clases del modelo')
            c1, c2 = st.columns(2)
            with c1:
                st.info('**Clase 0** — Sin Enfermedad Cardíaca')
            with c2:
                st.warning('**Clase 1** — Con Enfermedad Cardíaca')
 
else:
    st.subheader('📂 Carga de Datos por Lotes')
    st.info('📌 El CSV debe tener las columnas del dataset Cleveland. Puede incluir "target" para ver métricas.')
 
    uploaded_file = st.file_uploader('Sube tu CSV (hasta 200MB)', type=['csv'])
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        st.markdown('#### 👀 Vista previa')
        st.dataframe(df_raw.head(), use_container_width=True)
        st.caption(f'Total de registros: {len(df_raw)}')
 
        df_clean, constant_cols = clean_batch_data(df_raw)
 
        if constant_cols:
            st.info(f'Se eliminaron columnas constantes: {constant_cols}')
 
        missing_features = [f for f in FEATURE_COLUMNS if f not in df_clean.columns]
        if missing_features:
            st.error(f'Faltan columnas necesarias para el modelo: {missing_features}')
        else:
            X = df_clean[FEATURE_COLUMNS].copy()
            X_scaled = scaler.transform(X)
            report_items = []
            predictions = get_prediction_results(X_scaled, modelo_seleccionado)
 
            for model_name, pred_array, prob_array in predictions:
                st.markdown('---')
                st.markdown(f'#### 📊 Resultados - {model_name}')
                output = df_clean.copy()
                output[f'pred_{model_name.replace(" ", "_").lower()}'] = pred_array
                output[f'prob_{model_name.replace(" ", "_").lower()}'] = prob_array
 
                st.markdown('#### 🏷️ Clases del modelo')
                c1, c2 = st.columns(2)
                with c1:
                    st.info('**Clase 0** — Sin Enfermedad Cardíaca')
                with c2:
                    st.warning('**Clase 1** — Con Enfermedad Cardíaca')
 
                counts = pd.Series(pred_array).value_counts().rename({0: 'Sin enfermedad (0)', 1: 'Con enfermedad (1)'})
                st.bar_chart(counts)
 
                if 'target' in df_clean.columns:
                    st.markdown('**Reporte de Clasificación:**')
                    reporte = classification_report(df_clean['target'], pred_array,
                                                   target_names=['Sin Enfermedad (0)', 'Con Enfermedad (1)'])
                    st.text(reporte)
                    fig, ax = plt.subplots(figsize=(5, 3))
                    cm = confusion_matrix(df_clean['target'], pred_array, labels=[0, 1])
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues' if 'Logística' in model_name else 'Oranges',
                                ax=ax, linewidths=0.5,
                                xticklabels=['Sin Enfermedad (0)', 'Con Enfermedad (1)'],
                                yticklabels=['Sin Enfermedad (0)', 'Con Enfermedad (1)'])
                    ax.set_title(f'Matriz de Confusión - {model_name}', fontsize=11)
                    ax.set_xlabel('Predicho')
                    ax.set_ylabel('Real')
                    st.pyplot(fig)
 
                report_items.append((model_name, output))
 
            if report_items:
                name, download_df = report_items[-1]
                csv = download_df.to_csv(index=False).encode('utf-8')
                pdf = build_batch_pdf_report(download_df, name, 'target' in df_clean.columns)
                st.download_button('📥 Descargar resultados del lote', csv,
                                   file_name=f'resultados_lote_{name.replace(" ", "_")}.csv',
                                   mime='text/csv')
                st.download_button('📄 Descargar resultados por lote en PDF', pdf,
                                   file_name=f'resultados_lote_{name.replace(" ", "_")}.pdf',
                                   mime='application/pdf')
 
    st.markdown('---')
 
st.markdown("<p style='text-align: center; color: gray;'>Desarrollado por Rafael Polo Henao | Universidad Cooperativa de Colombia | 2026</p>", unsafe_allow_html=True)
           
