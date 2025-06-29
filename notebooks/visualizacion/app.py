#codestrike

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import traceback
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# ============= CONFIGURACIÓN MEJORADA DE RUTAS =============
def setup_module_paths():
    """Configurar rutas de módulos de forma más robusta"""
    possible_paths = [
        '../modelos/lgbm',  # Desde visualizacion/
        './modelos/lgbm',   # Desde raíz
        'modelos/lgbm',     # Relativa simple
        '.',                # Directorio actual
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and abs_path not in sys.path:
            sys.path.insert(0, abs_path)
            print(f"✅ Ruta agregada: {abs_path}")
    
    return True

# Configurar rutas
setup_module_paths()

# ============= IMPORTACIONES CON MANEJO DE ERRORES =============
try:
    import pred_lgbm as pred
    print("✅ Módulo 'pred_lgbm' importado correctamente")
except ImportError:
    try:
        import pred
        print("✅ Módulo 'pred' importado correctamente")
    except ImportError as e:
        st.error(f"❌ Error importando módulo de predicción: {e}")
        st.error("Verifica que tengas pred_lgbm.py en la ruta correcta")
        st.stop()

try:
    import funciones_lgbm as f_lgbm
    print("✅ Módulo 'funciones_lgbm' importado correctamente")
except ImportError as e:
    print(f"⚠️ Módulo 'funciones_lgbm' no encontrado: {e}")
    f_lgbm = None

# Configuración de la página
st.set_page_config(
    page_title="💰 Predictor de Salarios",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        font-size: 3rem;
        font-weight: bold;
    }
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #17becf;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_model_safe(uploaded_file):
    
    
    saved_file_path = None
    try:
        file_name = uploaded_file.name
        file_size = len(uploaded_file.getbuffer())
        
        print(f"📁 Archivo: {file_name}")
        print(f"📏 Tamaño: {file_size:,} bytes")
        
        # Verificar que el archivo no esté vacío
        if file_size == 0:
            return None, "Error: El archivo está vacío"
        
        # Guardar en el directorio actual con el nombre original
        saved_file_path = os.path.join(os.getcwd(), file_name)
        
        print(f"💾 Guardando en: {saved_file_path}")
        
        with open(saved_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        print(f"✅ Archivo guardado exitosamente")
        
        # Cargar exactamente igual que tu test exitoso
        print("🔄 Cargando modelo con joblib...")
        model_package = joblib.load(saved_file_path)
        print("✅ Modelo cargado exitosamente")
        
        # Verificar estructura básica
        required_keys = ['model']
        missing_keys = [key for key in required_keys if key not in model_package]
        
        if missing_keys:
            return None, f"Error: Claves faltantes en el modelo: {missing_keys}"
        
        # Limpiar archivo después de cargar
        try:
            os.remove(saved_file_path)
            print(f"🗑️ Archivo {saved_file_path} eliminado")
        except:
            pass
        
        return model_package, f"Modelo '{file_name}' cargado exitosamente"
            
    except Exception as e:
        error_msg = f"Error cargando modelo: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Limpiar archivo en caso de error
        if saved_file_path and os.path.exists(saved_file_path):
            try:
                os.remove(saved_file_path)
            except:
                pass
        
        return None, error_msg

def test_prediction_streamlit(model_package):
    """Test de predicción"""
    print("🧪 Testing predicción con un solo registro...")
    
    try:
        # Calcular grupos
        exp_group, age_group = pred.calculate_groups(
            age=60, 
            years_of_experience=24, 
            grouping_info=model_package.get('grouping_info')
        )
        
        # Crear registro de prueba
        test_record = pd.DataFrame({
            'Age': [60],
            'Gender': ['Male'],
            'Education_Level': ["PhD"],
            'Job_Title': ['CEO'],
            'Years_of_Experience': [24],
            'Description': ['I work with machine learning models and data analysis'],
            'Exp_group': [exp_group],
            'Age_group': [age_group]
        })
        
        # Predicción
        prediction = pred.predict(test_record, model_package)
        print(f"✅ Test exitoso: Predicción = ${prediction:,.2f}")
        return True, prediction
        
    except Exception as e:
        print(f"❌ Test falló: {e}")
        print(traceback.format_exc())
        return False, str(e)

def load_model_from_path(file_path):
    """Cargar modelo desde ruta"""
    try:
        print(f"🔄 Cargando modelo desde: {file_path}")
        model_package = joblib.load(file_path)
        print("✅ Modelo cargado exitosamente")
        
        # Test automático
        test_success, test_result = test_prediction_streamlit(model_package)
        if test_success:
            print(f"🎯 Test de predicción exitoso: ${test_result:,.2f}")
        else:
            print(f"⚠️ Test de predicción falló: {test_result}")
        
        return model_package
        
    except Exception as e:
        print(f"❌ Error: {e}")
        st.error(f"Error cargando modelo: {e}")
        return None

def display_model_info(model_package):
    
    """Mostrar información del modelo"""
    st.sidebar.markdown("### 📊 Información del Modelo")
    
    # Métricas principales
    if 'metrics' in model_package:
        metrics = model_package['metrics']
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if 'r2' in metrics:
                st.metric("R² Score", f"{metrics['r2']:.3f}")
            if 'rmse' in metrics:
                st.metric("RMSE", f"${metrics['rmse']:,.0f}")
        
        with col2:
            if 'mae' in metrics:
                st.metric("MAE", f"${metrics['mae']:,.0f}")
            if 'cv_rmse' in metrics:
                st.metric("CV RMSE", f"${metrics['cv_rmse']:,.0f}")
    
    # Información técnica
    st.sidebar.markdown("### 🔧 Detalles Técnicos")
    st.sidebar.info(f"""
    **Features:** {model_package.get('total_features', 'N/A')}
    **Categorías de Trabajo:** {len(model_package.get('job_categories', []))}
    **Niveles de Seniority:** {len(model_package.get('seniority_categories', []))}
    **Features Estadísticos:** {'✅' if model_package.get('has_statistical_features') else '❌'}
    """)

def show_grouping_rules(model_package):
    """Mostrar reglas de agrupación"""
    with st.expander("📋 Ver Reglas de Agrupación"):
        grouping_info = model_package.get('grouping_info', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**👥 Grupos de Edad:**")
            if 'age_bins' in grouping_info and 'age_labels' in grouping_info:
                age_bins = grouping_info['age_bins']
                age_labels = grouping_info['age_labels']
                
                for i, label in enumerate(age_labels):
                    if i < len(age_bins) - 1:
                        start = age_bins[i]
                        end = age_bins[i + 1] if age_bins[i + 1] != float('inf') else "+"
                        if end == "+":
                            st.write(f"• **{label}:** {start}+ años")
                        else:
                            st.write(f"• **{label}:** {start}-{end-1} años")
            else:
                st.write("• **Joven:** 18-29 años")
                st.write("• **Medio:** 30-37 años")
                st.write("• **Adulto:** 38-44 años")
                st.write("• **Senior:** 45+ años")
        
        with col2:
            st.markdown("**💼 Grupos de Experiencia:**")
            if 'exp_bins' in grouping_info and 'exp_labels' in grouping_info:
                exp_bins = grouping_info['exp_bins']
                exp_labels = grouping_info['exp_labels']
                
                for i, label in enumerate(exp_labels):
                    if i < len(exp_bins) - 1:
                        start = exp_bins[i]
                        end = exp_bins[i + 1] if exp_bins[i + 1] != float('inf') else "+"
                        if end == "+":
                            st.write(f"• **{label}:** {start}+ años")
                        else:
                            st.write(f"• **{label}:** {start}-{end-1} años")
            else:
                st.write("• **Junior:** 0-4 años")
                st.write("• **Medio:** 5-14 años")
                st.write("• **Senior:** 15+ años")

def create_user_input_form():
    
    """Crear formulario de entrada"""
    st.markdown("## 📝 Datos para la Predicción")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Información Personal")
        age = st.slider("Edad", min_value=18, max_value=70, value=30)
        gender = st.selectbox("Género", options=["Male", "Female"])
        education = st.selectbox("Nivel de Educación", 
                                options=["Bachelor's", "Master's", "PhD"], index=0)
    
    with col2:
        st.markdown("### 💼 Experiencia Profesional")
        years_of_experience = st.slider("Años de Experiencia", 
                                       min_value=0, max_value=40, value=5)
        job_title = st.text_input("Título del Trabajo", value="Software Engineer")
    
    st.markdown("### 📄 Descripción del Trabajo")
    description = st.text_area("Descripción de Responsabilidades",
        value="Responsible for developing software solutions and analyzing data.",
        height=120)
    
    return {
        'Age': age,
        'Gender': gender,
        'Education_Level': education,
        'Job_Title': job_title,
        'Years_of_Experience': years_of_experience,
        'Description': description
    }

def show_prediction_result(prediction, confidence_interval=None):
    """Mostrar resultado de predicción"""
    
    # Resultado principal
    st.markdown(f"""
    <div class="prediction-result">
        💰 Salario Estimado: ${prediction:,.0f}
    </div>
    """, unsafe_allow_html=True)
    
    # Métricas en columnas
    if confidence_interval:
        lower_bound = prediction - confidence_interval
        upper_bound = prediction + confidence_interval
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("💰 Predicción", f"${prediction:,.0f}")
        
        with col2:
            st.metric("📉 Mínimo Estimado", f"${max(0, lower_bound):,.0f}",
                     delta=f"-${confidence_interval:,.0f}")
        
        with col3:
            st.metric("📈 Máximo Estimado", f"${upper_bound:,.0f}",
                     delta=f"+${confidence_interval:,.0f}")
    
    # Interpretación
    st.markdown("### 💡 Interpretación del Resultado")
    
    if prediction > 150000:
        st.success("🔥 **Salario Premium** - Perfil altamente competitivo")
    elif prediction > 100000:
        st.success("✅ **Salario Alto** - Perfil muy valorado")
    elif prediction > 70000:
        st.info("📊 **Salario Competitivo** - Buen perfil")
    elif prediction > 50000:
        st.warning("📈 **Salario Medio** - Perfil en desarrollo")
    else:
        st.info("🌱 **Salario Inicial** - Perfil junior")

def show_profile_analysis(user_data, model_package):
    """Mostrar análisis del perfil"""
    
    # Calcular grupos
    exp_group, age_group = pred.calculate_groups(
        age=user_data['Age'],
        years_of_experience=user_data['Years_of_Experience'],
        grouping_info=model_package.get('grouping_info')
    )
    
    st.markdown("### 📊 Análisis de tu Perfil")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="info-box">
        <h4>👥 Grupo de Edad</h4>
        <p><strong>{age_group}</strong></p>
        <p>{user_data['Age']} años</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
        <h4>💼 Nivel de Experiencia</h4>
        <p><strong>{exp_group}</strong></p>
        <p>{user_data['Years_of_Experience']} años</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Ratio experiencia/edad
        exp_ratio = user_data['Years_of_Experience'] / user_data['Age']
        if exp_ratio > 0.6:
            exp_level, exp_color = "Muy Alta", "🔥"
        elif exp_ratio > 0.4:
            exp_level, exp_color = "Alta", "⚡"
        elif exp_ratio > 0.2:
            exp_level, exp_color = "Media", "📊"
        else:
            exp_level, exp_color = "Inicial", "🌱"
        
        st.markdown(f"""
        <div class="info-box">
        <h4>📈 Densidad de Experiencia</h4>
        <p><strong>{exp_color} {exp_level}</strong></p>
        <p>{exp_ratio:.1%} exp/edad</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Título principal
    st.markdown('<h1 class="main-header">💰 Predictor de Salarios</h1>', 
                unsafe_allow_html=True)
    
    # Session state
    if 'model_package' not in st.session_state:
        st.session_state.model_package = None
    
    # Sidebar para cargar modelo
    with st.sidebar:
        st.markdown("## ⚙️ Configuración del Modelo")
        
        # Opción 1: Subir archivo
        st.markdown("### 📁 Opción 1: Subir Archivo")
        uploaded_file = st.file_uploader("Subir Modelo (.pkl)", type=['pkl'])
        
        if uploaded_file is not None:
            if st.button("🔄 Cargar Modelo", type="primary"):
                with st.spinner("Cargando modelo..."):
                    model_package, message = load_model_safe(uploaded_file)
                    
                    if model_package:
                        st.session_state.model_package = model_package
                        st.success(message)
                        st.balloons()
                    else:
                        st.error(message)
        
        # Mostrar info del modelo
        if st.session_state.model_package:
            st.success("✅ Modelo Activo")
            display_model_info(st.session_state.model_package)
            
            # Test
            st.markdown("### 🧪 Verificación")
            if st.button("🔬 Ejecutar Test"):
                with st.spinner("Ejecutando test..."):
                    test_success, test_result = test_prediction_streamlit(
                        st.session_state.model_package)
                    
                    if test_success:
                        st.success(f"✅ Test exitoso: ${test_result:,.2f}")
                    else:
                        st.error(f"❌ Test falló: {test_result}")
        else:
            st.warning("⚠️ No hay modelo cargado")
    
    # Contenido principal
    if not st.session_state.model_package:
        st.markdown("""
        ## 👋 Bienvenido al Predictor de Salarios
        
        - 📊 **92+ características** técnicas y estadísticas
        - 🎯 **Modelo LightGBM** optimizado con Optuna
        - 📈 **Features estadísticos** comparativos por grupos
        - 🔍 **Análisis contextual** del mercado laboral
        
        ### 🚀 Para comenzar:
        1. Sube tu modelo entrenado (.pkl) en la barra lateral
        2. Completa la información del candidato
        3. Obtén predicciones precisas y análisis detallado
        """)
        return
    
    # Mostrar reglas de agrupación
    show_grouping_rules(st.session_state.model_package)
    
    # Formulario de entrada
    user_data = create_user_input_form()
    
    # Análisis del perfil
    show_profile_analysis(user_data, st.session_state.model_package)
    
    # Botón de predicción
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Predecir Salario", type="primary", use_container_width=True):
            
            with st.spinner("🧠 Analizando perfil y calculando predicción..."):
                try:
                    # Calcular grupos
                    exp_group, age_group = pred.calculate_groups(
                        age=user_data['Age'],
                        years_of_experience=user_data['Years_of_Experience'],
                        grouping_info=st.session_state.model_package.get('grouping_info')
                    )
                    
                    # Crear DataFrame
                    prediction_data = pd.DataFrame({
                        'Age': [user_data['Age']],
                        'Gender': [user_data['Gender']],
                        'Education_Level': [user_data['Education_Level']],
                        'Job_Title': [user_data['Job_Title']],
                        'Years_of_Experience': [user_data['Years_of_Experience']],
                        'Description': [user_data['Description']],
                        'Exp_group': [exp_group],
                        'Age_group': [age_group]
                    })
                    
                    # Info debug
                    st.info(f"🔍 Grupos calculados: {age_group} (edad), {exp_group} (experiencia)")
                    
                    # Predicción
                    prediction = pred.predict(prediction_data, st.session_state.model_package)
                    
                    # Intervalo de confianza
                    metrics = st.session_state.model_package.get('metrics', {})
                    confidence_interval = metrics.get('rmse', prediction * 0.15)
                    
                    # Mostrar resultados
                    show_prediction_result(prediction, confidence_interval)
                    
                    # Detalles adicionales
                    with st.expander("📋 Detalles de la Predicción"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📝 Datos Procesados:**")
                            st.write(f"• **Edad:** {user_data['Age']} años → {age_group}")
                            st.write(f"• **Experiencia:** {user_data['Years_of_Experience']} años → {exp_group}")
                            st.write(f"• **Educación:** {user_data['Education_Level']}")
                            st.write(f"• **Género:** {user_data['Gender']}")
                            st.write(f"• **Puesto:** {user_data['Job_Title']}")
                        
                        with col2:
                            st.markdown("**🔧 Información Técnica:**")
                            model_info = st.session_state.model_package
                            st.write(f"• **Features utilizadas:** {model_info.get('total_features', 'N/A')}")
                            st.write(f"• **Modelo:** {model_info.get('model_name', 'LightGBM')}")
                            if 'rmse' in metrics:
                                st.write(f"• **RMSE del modelo:** ${metrics['rmse']:,.0f}")
                            if 'r2' in metrics:
                                st.write(f"• **R² Score:** {metrics['r2']:.3f}")
                    
                except Exception as e:
                    st.error(f"❌ Error en la predicción: {str(e)}")
                    
                    with st.expander("🔧 Información de Debug"):
                        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()