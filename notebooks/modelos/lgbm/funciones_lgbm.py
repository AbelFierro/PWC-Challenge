import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
#import xgboost as xgb
import lightgbm as lgb
import optuna
#from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')




################ Guardar modelo optimizado        ########################################

def save_with_stats(model_results, filename="salary_model_complete_with_stats.pkl"):
    """
    Guardar modelo completo incluyendo features estadísticos
    """
    import joblib
    
    print("💾 Guardando modelo completo con features estadísticos...")
    
    # Extraer componentes
    lgb_results = model_results['model_results']['LightGBM_Optuna']
    model = lgb_results['model']
    metrics = {k: v for k, v in lgb_results.items() if k != 'model'}
    
    # Crear paquete completo
    complete_package = {
        'model': model,
        'model_name': 'LightGBM_Optuna_WithStats',
        'feature_names': model_results['feature_names'],
        'job_categories': model_results['job_categories'],
        'seniority_categories': model_results['seniority_categories'],
        'stats_dict': model_results['stats_dict'],  # ¡NUEVO!
        'grouping_info': model_results['grouping_info'],
        'metrics': metrics,
        'total_features': len(model_results['feature_names']),
        'training_data_shape': model_results['X_train'].shape,
        'has_statistical_features': True  # Flag para identificar
    }
    
    # Guardar
    joblib.dump(complete_package, filename)
    
    print(f"✅ Modelo completo guardado en {filename}")
    print(f"📦 Incluye:")
    print(f"   🤖 Modelo LightGBM optimizado")
    print(f"   🔢 {len(model_results['feature_names'])} características")
    print(f"   📊 Features estadísticos (stats_dict)")
    print(f"   🏷️  {len(model_results['job_categories'])} categorías de trabajo")
    print(f"   👔 {len(model_results['seniority_categories'])} niveles de seniority")
    print(f"   📈 Información de agrupación")
    print(f"   📉 Métricas del modelo")
    
    return complete_package


################ Funciones de Análisis de modelos ########################################

def get_feature_importance(model, feature_names=None, top_n=10):
    """
    Obtener y mostrar la importancia de las características
    """
    if not hasattr(model, 'feature_importances_'):
        print("El modelo no tiene importancia de características disponible")
        return None
    
    importances = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(importances))]
    
    # Crear DataFrame con importancias
    import pandas as pd
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top {top_n} características más importantes:")
    for i, (_, row) in enumerate(feature_importance_df.head(top_n).iterrows()):
        print(f"   {i+1:2d}. {row['feature']:20s} - {row['importance']:.4f}")
    
    return feature_importance_df


def analyze_optuna_optimization(study, top_n=10):
    """
    Analiza los resultados de una optimización de Optuna.

    Args:
        study (optuna.Study): El estudio de Optuna ya ejecutado.
        top_n (int): Número de hiperparámetros más importantes a mostrar.

    Returns:
        Tuple: (mejores parámetros encontrados, mejor valor)
    """

    print(f"🔬 Análisis de optimización Optuna:")
    print(f"   Número total de trials: {len(study.trials)}")
    print(f"   Mejor valor: ${study.best_value:,.2f}")
    print(f"   Trials completados: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"   Trials fallidos: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

    # Importancia de hiperparámetros
    try:
        importance = optuna.importance.get_param_importances(study)
        print(f"   🔍 Importancia de hiperparámetros (Top {min(top_n, len(importance))}):")
        for i, (param, imp) in enumerate(list(importance.items())[:top_n]):
            print(f"      {i+1:2d}. {param:20s} - {imp:.4f}")
    except Exception as e:
        print(f"   ⚠️  No se pudo calcular la importancia de hiperparámetros: {e}")

    return study.best_params, study.best_value


################## Funciones de agrupamiento #########################################################

def create_and_save_grouping_info(data):
    """
    Crear los grupos y guardar la información necesaria para aplicarlos a nuevos registros
    """
    print("📊 Creando grupos y guardando información de rangos...")
    
    # ============= CREAR EXP_GROUP Y GUARDAR RANGOS =============
    # Usar cut para experiencia (rangos fijos)
    exp_bins = [0, 5, 15, 40]
    exp_labels = ['Junior', 'Medio', 'Senior']
    data['Exp_group'] = pd.cut(data['Years_of_Experience'], bins=exp_bins, labels=exp_labels)
    
    # ============= CREAR AGE_GROUP Y GUARDAR CUANTILES =============
    # Usar qcut para edad (cuantiles basados en datos)
    #age_qcut_result = pd.qcut(data['Age'], q=4, labels=['Joven', 'Adulto', 'Mediano', 'Senior'], retbins=True)
    #data['Age_group'] = age_qcut_result[0]
    #age_bins = age_qcut_result[1]  # Los rangos calculados por qcut
    
    
    age_bins = [18, 30, 38, 45, float('inf')]
    age_labels = ['Joven','Medio','Adulto', 'Senior']
    data['Age_group'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, right=True, include_lowest=True)
    
    # ============= GUARDAR INFORMACIÓN DE AGRUPACIÓN =============
    grouping_info = {
        # Para Exp_group (rangos fijos)
        'exp_bins': exp_bins,
        'exp_labels': exp_labels,
        
        # Para Age_group (cuantiles calculados)
        'age_bins': age_bins,
        'age_labels': age_labels,
        
        # Estadísticas adicionales
        'age_stats': {
            'min': float(data['Age'].min()),
            'max': float(data['Age'].max()),
            'mean': float(data['Age'].mean()),
            'std': float(data['Age'].std())
        },
        'exp_stats': {
            'min': float(data['Years_of_Experience'].min()),
            'max': float(data['Years_of_Experience'].max()),
            'mean': float(data['Years_of_Experience'].mean()),
            'std': float(data['Years_of_Experience'].std())
        }
    }
    
    #print(f"✅ Grupos creados:")
    #print(f"   📅 Age_group - Rangos: {[f'{age_bins[i]:.1f}-{age_bins[i+1]:.1f}' for i in range(len(age_bins)-1)]}")
    #print(f"   💼 Exp_group - Rangos: Junior(0-5), Medio(5-15), Senior(15+)")
    #print(f"   📊 Distribución Age_group: {data['Age_group'].value_counts().to_dict()}")
    #print(f"   📊 Distribución Exp_group: {data['Exp_group'].value_counts().to_dict()}")
    
    return data, grouping_info





def get_all_categories(train_data):
    """
    Obtener todas las categorías únicas del dataset de entrenamiento
    para garantizar consistencia en train/test
    """
    job_title_lower = train_data['Job_Title'].str.lower().fillna('')
    
    def assign_job_category(title):
        title = title.lower()
        if any(word in title for word in ['director', 'head', 'chief', 'vp']):
            return 'EXECUTIVE'
        elif any(word in title for word in ['manager', 'lead', 'supervisor']):
            return 'MANAGEMENT'
        elif any(word in title for word in ['engineer', 'developer', 'programmer']):
            return 'ENGINEERING'
        elif any(word in title for word in ['scientist', 'data', 'machine learning']):
            return 'DATA_SCIENCE'
        elif any(word in title for word in ['analyst', 'research']):
            return 'ANALYST'
        elif any(word in title for word in ['sales', 'account']):
            return 'SALES'
        elif any(word in title for word in ['marketing', 'brand']):
            return 'MARKETING'
        elif any(word in title for word in ['financial', 'finance', 'accounting']):
            return 'FINANCE'
        elif any(word in title for word in ['operations', 'coordinator', 'administrator']):
            return 'OPERATIONS'
        elif any(word in title for word in ['designer', 'design', 'creative']):
            return 'DESIGN'
        elif any(word in title for word in ['consultant', 'consulting']):
            return 'CONSULTING'
        else:
            return 'OTHER'
    
    def assign_seniority_level(title):
        title = title.lower()
        if any(word in title for word in ['director', 'head', 'chief', 'vp']):
            return 'EXECUTIVE'
        elif any(word in title for word in ['senior', 'lead', 'principal']):
            return 'SENIOR'
        elif any(word in title for word in ['manager', 'supervisor']):
            return 'MANAGER'
        elif any(word in title for word in ['junior', 'entry', 'intern']):
            return 'JUNIOR'
        else:
            return 'MID'
    
    job_categories = job_title_lower.apply(assign_job_category).unique()
    seniority_levels = job_title_lower.apply(assign_seniority_level).unique()
    
    return sorted(job_categories), sorted(seniority_levels)



















########################################################################################################

##################     Funciones de FE       ###########################################################

#Funciones sin dataleakage

def create_features(data, all_job_categories=None, all_seniority_levels=None):
    """
    Función unificada que combina todas las features:
    - Features originales existentes
    - Nuevos ratios sugeridos  
    - Agrupaciones inteligentes de job titles
    - Features de texto mejoradas
    
    Parameters:
    - data: DataFrame con los datos
    - all_job_categories: lista de todas las categorías posibles (para consistencia)
    - all_seniority_levels: lista de todos los niveles de seniority posibles
    """
    print("🔧 Creando todas las características mejoradas...")
    
    features = pd.DataFrame()
    
    # ============= CARACTERÍSTICAS BÁSICAS ORIGINALES =============
    
    features['age'] = data['Age']
    features['years_experience'] = data['Years_of_Experience']
    features['age_experience_ratio'] = data['Age'] / (data['Years_of_Experience'] + 1)
    features['experience_squared'] = data['Years_of_Experience'] ** 2
    features['age_exp_interaction'] = data['Age'] * data['Years_of_Experience']
    features['senior_pro'] = ((data['Age'] > 45) & (data['Years_of_Experience'] > 15)).astype(int)
    #features['Age_group'] = data['Age_group'].astype('category')
    
    
    #Viendo la proximidad de grupo mediano y senior en edades y los problemas de prediccion probemos esto
    
    #features['Age_group'] = data['grupo_etario'].apply(
    #lambda x: 'Alta experiencia' if x in ['Mediano', 'Senior'] else x
    #)
    
    edu_map = {"Bachelor's": 1, "Master's": 2, "PhD": 3}
    features['Education_Level_Num'] = data['Education_Level'].map(edu_map)
    
    features['start_year'] = 2025 - data['Years_of_Experience']
    features['age_edu'] = data['Age'] * features['Education_Level_Num']
    features['exp_edu'] = data['Years_of_Experience'] * features['Education_Level_Num']
    
    # ============= NUEVOS RATIOS DE MADUREZ PROFESIONAL =============
    features['experience_age_ratio'] = data['Years_of_Experience'] / data['Age']
    features['career_start_age'] = data['Age'] - data['Years_of_Experience']
    features['career_maturity'] = np.where(data['Years_of_Experience'] >= 10, 1, 0)
    
    # ============= CARACTERÍSTICAS CATEGÓRICAS ORIGINALES =============
    # Género
    features['gender_male'] = (data['Gender'] == 'Male').astype(int)
    features['gender_female'] = (data['Gender'] == 'Female').astype(int)
    
    # Educación (originales)
    features['edu_bachelors'] = (data['Education_Level'] == "Bachelor's").astype(int)
    features['edu_masters'] = (data['Education_Level'] == "Master's").astype(int) 
    features['edu_phd'] = (data['Education_Level'] == 'PhD').astype(int)
    
    # Educación (nuevas)
    features['has_graduate_degree'] = (
        (data['Education_Level'] == "Master's") | 
        (data['Education_Level'] == 'PhD')
    ).astype(int)
    
    # ============= JOB TITLES - CARACTERÍSTICAS ORIGINALES =============
    job_title_lower = data['Job_Title'].str.lower().fillna('')
    
    # Variables originales de job titles
    features['job_engineer'] = job_title_lower.str.contains('engineer', na=False).astype(int)
    features['job_manager'] = job_title_lower.str.contains('manager', na=False).astype(int)
    features['job_analyst'] = job_title_lower.str.contains('analyst', na=False).astype(int)
    features['job_scientist'] = job_title_lower.str.contains('scientist', na=False).astype(int)
    features['job_senior'] = job_title_lower.str.contains('senior', na=False).astype(int)
    features['job_data'] = job_title_lower.str.contains('data', na=False).astype(int)
    
    executive_titles = ['ceo', 'chief', 'director', 'president', 'founder']
    features['is_exec'] = job_title_lower.apply(
    lambda x: int(any(title in x for title in executive_titles))
    )
    features['age_exec'] = data['Age'] * features['is_exec']
    features['exp_exec'] = data['Years_of_Experience'] * features['is_exec']
    
    # el outlier 250000 tiene el titulo de ceo
    # solo un caso deberia haber mas ejemplos para replicar
    
    features['is_ceo'] = job_title_lower.str.contains('ceo').astype(int)
    features['ceo_edu'] = features['is_ceo'] * features['Education_Level_Num']
    
    # ============= NUEVAS AGRUPACIONES DE JOB TITLES =============
    
    def assign_job_category(title):
        """Asignar categoría principal"""
        title = title.lower()
        if any(word in title for word in ['director', 'head', 'chief', 'vp']):
            return 'EXECUTIVE'
        elif any(word in title for word in ['manager', 'lead', 'supervisor']):
            return 'MANAGEMENT'
        elif any(word in title for word in ['engineer', 'developer', 'programmer']):
            return 'ENGINEERING'
        elif any(word in title for word in ['scientist', 'data', 'machine learning']):
            return 'DATA_SCIENCE'
        elif any(word in title for word in ['analyst', 'research']):
            return 'ANALYST'
        elif any(word in title for word in ['sales', 'account']):
            return 'SALES'
        elif any(word in title for word in ['marketing', 'brand']):
            return 'MARKETING'
        elif any(word in title for word in ['financial', 'finance', 'accounting']):
            return 'FINANCE'
        elif any(word in title for word in ['operations', 'coordinator', 'administrator']):
            return 'OPERATIONS'
        elif any(word in title for word in ['designer', 'design', 'creative']):
            return 'DESIGN'
        elif any(word in title for word in ['consultant', 'consulting']):
            return 'CONSULTING'
        else:
            return 'OTHER'
    
    def assign_seniority_level(title):
        """Asignar nivel de seniority"""
        title = title.lower()
        if any(word in title for word in ['director', 'head', 'chief', 'vp']):
            return 'EXECUTIVE'
        elif any(word in title for word in ['senior', 'lead', 'principal']):
            return 'SENIOR'
        elif any(word in title for word in ['manager', 'supervisor']):
            return 'MANAGER'
        elif any(word in title for word in ['junior', 'entry', 'intern']):
            return 'JUNIOR'
        else:
            return 'MID'
    
    # Aplicar categorizaciones
    job_categories = job_title_lower.apply(assign_job_category)
    seniority_levels = job_title_lower.apply(assign_seniority_level)
    
    # ============= CREAR DUMMIES CON TODAS LAS CATEGORÍAS =============
    
    # Definir todas las categorías posibles si no se proporcionan
    if all_job_categories is None:
        all_job_categories = ['ANALYST', 'CONSULTING', 'DATA_SCIENCE', 'DESIGN', 'ENGINEERING', 
                             'EXECUTIVE', 'FINANCE', 'MANAGEMENT', 'MARKETING', 'OPERATIONS', 
                             'OTHER', 'SALES']
    
    if all_seniority_levels is None:
        all_seniority_levels = ['EXECUTIVE', 'JUNIOR', 'MANAGER', 'MID', 'SENIOR']
    
    # Crear dummies para categorías funcionales - GARANTIZANDO TODAS LAS COLUMNAS
    for category in all_job_categories:
        features[f'job_cat_{category}'] = (job_categories == category).astype(int)
    
    # Crear dummies para seniority - GARANTIZANDO TODAS LAS COLUMNAS
    for level in all_seniority_levels:
        features[f'seniority_{level}'] = (seniority_levels == level).astype(int)
    
    # Variables binarias adicionales
    features['is_tech_role'] = job_title_lower.apply(
        lambda x: any(keyword in x for keyword in ['engineer', 'developer', 'scientist', 'data', 'technical'])
    ).astype(int)
    
    features['is_management_role'] = (seniority_levels.isin(['MANAGER', 'EXECUTIVE'])).astype(int)
    features['is_senior_role'] = (seniority_levels.isin(['SENIOR', 'EXECUTIVE'])).astype(int)
    
    # ============= FEATURES DE TEXTO DE DESCRIPTION =============
    if 'Description' in data.columns:
        # Contar términos de liderazgo
        leadership_terms = ['lead', 'manage', 'oversee', 'supervise', 'mentor', 'team', 'strategic']
        features['leadership_terms_count'] = data['Description'].str.lower().str.count('|'.join(leadership_terms))
        
        # Contar términos técnicos
        tech_terms = ['python', 'java', 'sql', 'machine learning', 'analytics', 'programming']
        features['tech_terms_count'] = data['Description'].str.lower().str.count('|'.join(tech_terms))
        
        # Métricas de texto
        features['word_count'] = data['Description'].str.split().str.len()
        features['avg_word_length'] = data['Description'].str.len() / features['word_count']
        
        # Términos positivos
        positive_terms = ['excellent', 'passion', 'committed', 'dedicated', 'expertise']
        features['sentiment_subjectivity'] = data['Description'].str.lower().str.count('|'.join(positive_terms))
    else:
        # Si no hay columna Description, crear variables vacías
        features['leadership_terms_count'] = 0
        features['tech_terms_count'] = 0
        features['word_count'] = 0
        features['avg_word_length'] = 0
        features['sentiment_subjectivity'] = 0
    
    # ============= RATIOS AVANZADOS =============
    
    # Ratios de educación y experiencia
    features['education_experience_synergy'] = (
        features['edu_phd'] * np.log1p(data['Years_of_Experience'])
    )
    features['overqualified'] = np.where(
        (data['Education_Level'] == 'PhD') & (data['Years_of_Experience'] < 5), 1, 0
    )
    
    # Ratios de seniority y responsabilidad
    features['seniority_experience_ratio'] = features['job_senior'] * data['Years_of_Experience']
    features['leadership_potential'] = (
        features['is_management_role'] * (data['Years_of_Experience'] / data['Age'])
    )
    
    # ============= INTERACCIONES COMPLEJAS =============
    
    # Combos valiosos
    features['tech_senior_combo'] = features['is_tech_role'] * features['is_senior_role']
    features['tech_management_combo'] = features['is_tech_role'] * features['is_management_role']
    
    # ============= PREPARAR RESULTADO =============
    
    # Limpiar nombres de columnas (reemplazar espacios y caracteres especiales)
    features.columns = [col.replace(' ', '_').replace('(', '').replace(')', '') for col in features.columns]
    
    # Reemplazar NaN con 0
    features = features.fillna(0)
    
    # Obtener nombres de características
    feature_names = list(features.columns)
    
    print(f"✅ Creadas {len(feature_names)} características en total")
    print(f"   - Variables numéricas básicas: {len([c for c in feature_names if c in ['age', 'years_experience', 'age_experience_ratio']])}")
    print(f"   - Variables de educación: {len([c for c in feature_names if 'edu_' in c])}")
    print(f"   - Variables de job category: {len([c for c in feature_names if 'job_cat_' in c])}")
    print(f"   - Variables de seniority: {len([c for c in feature_names if 'seniority_' in c])}")
    print(f"   - Variables de texto: {len([c for c in feature_names if any(x in c for x in ['terms_count', 'word_'])])}")
    print(f"   - Ratios y scores: {len([c for c in feature_names if any(x in c for x in ['ratio', 'score', 'combo'])])}")
    
    return features, feature_names



############Features problemáticas###################################################################

#Posible dataleakage

def create_statistical_features(data, stats_dict=None, is_training=True):
    """
    Crear features estadísticos usando SOLO variables disponibles en producción
    (Sin usar Salary o cualquier variable target)
    
    Parameters:
    - data: DataFrame con los datos (SIN columna Salary)
    - stats_dict: Diccionario con estadísticas calculadas en train (None para training)
    - is_training: True si estamos en entrenamiento, False si en predicción
    
    Returns:
    - features_df: DataFrame con las nuevas features
    - stats_dict: Diccionario con estadísticas (solo si is_training=True)
    """
    
    print(f"📊 Creando features estadísticos para producción ({'TRAIN' if is_training else 'PREDICT'})...")
    
    # Verificar que no hay columna Salary
    if 'Salary' in data.columns:
        raise ValueError("❌ ERROR: 'Salary' encontrado en data. Remover antes de crear features estadísticos.")
    
    features = pd.DataFrame(index=data.index)
    
    # ============= FEATURES ESTADÍSTICOS POR GRUPOS =============
    
    if is_training:
        print("   🔄 Calculando estadísticas en TRAIN (solo variables de producción)...")
        stats_dict = {}
        
        # GRUPO 1: Estadísticas por Education_Level
        education_stats = data.groupby('Education_Level').agg({
            'Age': ['mean', 'median', 'std'],
            'Years_of_Experience': ['mean', 'median', 'std']
            # ✅ SIN Salary - no estará disponible en producción
        }).round(2)
        
        # Flatten columns
        education_stats.columns = [f"edu_{col[0]}_{col[1]}" for col in education_stats.columns]
        stats_dict['education_stats'] = education_stats.to_dict('index')
        
        # GRUPO 2: Estadísticas por Gender
        gender_stats = data.groupby('Gender').agg({
            'Age': ['mean', 'median', 'std'],
            'Years_of_Experience': ['mean', 'median', 'std']
            # ✅ SIN Salary
        }).round(2)
        
        gender_stats.columns = [f"gender_{col[0]}_{col[1]}" for col in gender_stats.columns]
        stats_dict['gender_stats'] = gender_stats.to_dict('index')
        
        # GRUPO 3: Estadísticas por Age_group
        if 'Age_group' in data.columns:
            age_group_stats = data.groupby('Age_group').agg({
                'Years_of_Experience': ['mean', 'median', 'std']
                # ✅ SIN Salary
            }).round(2)
            
            age_group_stats.columns = [f"age_grp_{col[0]}_{col[1]}" for col in age_group_stats.columns]
            stats_dict['age_group_stats'] = age_group_stats.to_dict('index')
        
        # GRUPO 4: Estadísticas por Exp_group
        if 'Exp_group' in data.columns:
            exp_group_stats = data.groupby('Exp_group').agg({
                'Age': ['mean', 'median', 'std']
                # ✅ SIN Salary
            }).round(2)
            
            exp_group_stats.columns = [f"exp_grp_{col[0]}_{col[1]}" for col in exp_group_stats.columns]
            stats_dict['exp_group_stats'] = exp_group_stats.to_dict('index')
        
        # GRUPO 5: Estadísticas por combinaciones (Education + Gender)
        combo_stats = data.groupby(['Education_Level', 'Gender']).agg({
            'Age': ['mean', 'std'],
            'Years_of_Experience': ['mean', 'std']
            # ✅ SIN Salary
        }).round(2)
        
        combo_stats.columns = [f"combo_{col[0]}_{col[1]}" for col in combo_stats.columns]
        stats_dict['combo_stats'] = combo_stats.to_dict('index')
        
        # GRUPO 6: Estadísticas por Job_Title (agregadas por categorías)
        if 'Job_Title' in data.columns:
            # Crear categorías de job title para estadísticas
            def get_job_category_for_stats(title):
                title = title.lower()
                if any(word in title for word in ['engineer', 'developer', 'programmer']):
                    return 'TECH'
                elif any(word in title for word in ['manager', 'director', 'lead']):
                    return 'MANAGEMENT'
                elif any(word in title for word in ['analyst', 'scientist', 'data']):
                    return 'ANALYTICS'
                elif any(word in title for word in ['sales', 'marketing', 'business']):
                    return 'BUSINESS'
                else:
                    return 'OTHER'
            
            data_temp = data.copy()
            data_temp['Job_Category'] = data_temp['Job_Title'].apply(get_job_category_for_stats)
            
            job_category_stats = data_temp.groupby('Job_Category').agg({
                'Age': ['mean', 'std'],
                'Years_of_Experience': ['mean', 'std']
                # ✅ SIN Salary
            }).round(2)
            
            job_category_stats.columns = [f"job_cat_{col[0]}_{col[1]}" for col in job_category_stats.columns]
            stats_dict['job_category_stats'] = job_category_stats.to_dict('index')
        
        # ESTADÍSTICAS GLOBALES (solo variables disponibles en producción)
        global_stats = {
            'age_global_mean': data['Age'].mean(),
            'age_global_std': data['Age'].std(),
            'age_global_median': data['Age'].median(),
            'exp_global_mean': data['Years_of_Experience'].mean(),
            'exp_global_std': data['Years_of_Experience'].std(),
            'exp_global_median': data['Years_of_Experience'].median(),
            'description_length_mean': data['Description'].str.len().mean() if 'Description' in data.columns else 0,
            'description_length_std': data['Description'].str.len().std() if 'Description' in data.columns else 0,
            'description_word_count_mean': data['Description'].str.split().str.len().mean() if 'Description' in data.columns else 0,
            'description_word_count_std': data['Description'].str.split().str.len().std() if 'Description' in data.columns else 0
        }
        stats_dict['global_stats'] = global_stats
        
        print(f"   ✅ Estadísticas calculadas para {len(stats_dict)} grupos")
    
    else:
        print("   📥 Usando estadísticas pre-calculadas de TRAIN...")
        if stats_dict is None:
            raise ValueError("stats_dict es requerido para predicción")
    
    # ============= APLICAR ESTADÍSTICAS A LOS DATOS =============
    
    # Features por Education_Level
    if 'education_stats' in stats_dict:
        for idx, row in data.iterrows():
            edu_level = row['Education_Level']
            if edu_level in stats_dict['education_stats']:
                edu_stats = stats_dict['education_stats'][edu_level]
                
                # Comparar con promedios de su grupo educativo
                features.loc[idx, 'age_vs_edu_mean'] = row['Age'] - edu_stats.get('edu_Age_mean', row['Age'])
                features.loc[idx, 'exp_vs_edu_mean'] = row['Years_of_Experience'] - edu_stats.get('edu_Years_of_Experience_mean', row['Years_of_Experience'])
                
                # Z-scores vs su grupo educativo
                edu_age_std = edu_stats.get('edu_Age_std', 1)
                edu_exp_std = edu_stats.get('edu_Years_of_Experience_std', 1)
                
                features.loc[idx, 'age_zscore_vs_edu'] = (row['Age'] - edu_stats.get('edu_Age_mean', row['Age'])) / max(edu_age_std, 0.1)
                features.loc[idx, 'exp_zscore_vs_edu'] = (row['Years_of_Experience'] - edu_stats.get('edu_Years_of_Experience_mean', row['Years_of_Experience'])) / max(edu_exp_std, 0.1)
                
                # Diferencias relativas
                edu_age_mean = edu_stats.get('edu_Age_mean', row['Age'])
                edu_exp_mean = edu_stats.get('edu_Years_of_Experience_mean', row['Years_of_Experience'])
                
                features.loc[idx, 'age_pct_vs_edu'] = (row['Age'] - edu_age_mean) / max(edu_age_mean, 1) if edu_age_mean > 0 else 0
                features.loc[idx, 'exp_pct_vs_edu'] = (row['Years_of_Experience'] - edu_exp_mean) / max(edu_exp_mean, 1) if edu_exp_mean > 0 else 0
                
            else:
                # Valores por defecto si no se encuentra el grupo
                features.loc[idx, 'age_vs_edu_mean'] = 0
                features.loc[idx, 'exp_vs_edu_mean'] = 0
                features.loc[idx, 'age_zscore_vs_edu'] = 0
                features.loc[idx, 'exp_zscore_vs_edu'] = 0
                features.loc[idx, 'age_pct_vs_edu'] = 0
                features.loc[idx, 'exp_pct_vs_edu'] = 0
    
    # Features por Gender
    if 'gender_stats' in stats_dict:
        for idx, row in data.iterrows():
            gender = row['Gender']
            if gender in stats_dict['gender_stats']:
                gender_stats = stats_dict['gender_stats'][gender]
                
                features.loc[idx, 'age_vs_gender_mean'] = row['Age'] - gender_stats.get('gender_Age_mean', row['Age'])
                features.loc[idx, 'exp_vs_gender_mean'] = row['Years_of_Experience'] - gender_stats.get('gender_Years_of_Experience_mean', row['Years_of_Experience'])
                
                # Z-scores vs género
                gender_age_std = gender_stats.get('gender_Age_std', 1)
                gender_exp_std = gender_stats.get('gender_Years_of_Experience_std', 1)
                
                features.loc[idx, 'age_zscore_vs_gender'] = (row['Age'] - gender_stats.get('gender_Age_mean', row['Age'])) / max(gender_age_std, 0.1)
                features.loc[idx, 'exp_zscore_vs_gender'] = (row['Years_of_Experience'] - gender_stats.get('gender_Years_of_Experience_mean', row['Years_of_Experience'])) / max(gender_exp_std, 0.1)
                
            else:
                features.loc[idx, 'age_vs_gender_mean'] = 0
                features.loc[idx, 'exp_vs_gender_mean'] = 0
                features.loc[idx, 'age_zscore_vs_gender'] = 0
                features.loc[idx, 'exp_zscore_vs_gender'] = 0
    
    # Features por Age_group (si existe)
    if 'Age_group' in data.columns and 'age_group_stats' in stats_dict:
        for idx, row in data.iterrows():
            age_group = row['Age_group']
            if age_group in stats_dict['age_group_stats']:
                age_grp_stats = stats_dict['age_group_stats'][age_group]
                
                features.loc[idx, 'exp_vs_age_group_mean'] = row['Years_of_Experience'] - age_grp_stats.get('age_grp_Years_of_Experience_mean', row['Years_of_Experience'])
                
                age_grp_exp_std = age_grp_stats.get('age_grp_Years_of_Experience_std', 1)
                features.loc[idx, 'exp_zscore_vs_age_group'] = (row['Years_of_Experience'] - age_grp_stats.get('age_grp_Years_of_Experience_mean', row['Years_of_Experience'])) / max(age_grp_exp_std, 0.1)
                
            else:
                features.loc[idx, 'exp_vs_age_group_mean'] = 0
                features.loc[idx, 'exp_zscore_vs_age_group'] = 0
    
    # Features por Exp_group (si existe)
    if 'Exp_group' in data.columns and 'exp_group_stats' in stats_dict:
        for idx, row in data.iterrows():
            exp_group = row['Exp_group']
            if exp_group in stats_dict['exp_group_stats']:
                exp_grp_stats = stats_dict['exp_group_stats'][exp_group]
                
                features.loc[idx, 'age_vs_exp_group_mean'] = row['Age'] - exp_grp_stats.get('exp_grp_Age_mean', row['Age'])
                
                exp_grp_age_std = exp_grp_stats.get('exp_grp_Age_std', 1)
                features.loc[idx, 'age_zscore_vs_exp_group'] = (row['Age'] - exp_grp_stats.get('exp_grp_Age_mean', row['Age'])) / max(exp_grp_age_std, 0.1)
                
            else:
                features.loc[idx, 'age_vs_exp_group_mean'] = 0
                features.loc[idx, 'age_zscore_vs_exp_group'] = 0
    
    # Features por combinaciones Education + Gender
    if 'combo_stats' in stats_dict:
        for idx, row in data.iterrows():
            combo_key = (row['Education_Level'], row['Gender'])
            if combo_key in stats_dict['combo_stats']:
                combo_stats = stats_dict['combo_stats'][combo_key]
                
                features.loc[idx, 'age_vs_edu_gender_mean'] = row['Age'] - combo_stats.get('combo_Age_mean', row['Age'])
                features.loc[idx, 'exp_vs_edu_gender_mean'] = row['Years_of_Experience'] - combo_stats.get('combo_Years_of_Experience_mean', row['Years_of_Experience'])
                
            else:
                features.loc[idx, 'age_vs_edu_gender_mean'] = 0
                features.loc[idx, 'exp_vs_edu_gender_mean'] = 0
    
    # Features por Job Category
    if 'job_category_stats' in stats_dict and 'Job_Title' in data.columns:
        def get_job_category_for_stats(title):
            title = title.lower()
            if any(word in title for word in ['engineer', 'developer', 'programmer']):
                return 'TECH'
            elif any(word in title for word in ['manager', 'director', 'lead']):
                return 'MANAGEMENT'
            elif any(word in title for word in ['analyst', 'scientist', 'data']):
                return 'ANALYTICS'
            elif any(word in title for word in ['sales', 'marketing', 'business']):
                return 'BUSINESS'
            else:
                return 'OTHER'
        
        for idx, row in data.iterrows():
            job_category = get_job_category_for_stats(row['Job_Title'])
            if job_category in stats_dict['job_category_stats']:
                job_stats = stats_dict['job_category_stats'][job_category]
                
                features.loc[idx, 'age_vs_job_cat_mean'] = row['Age'] - job_stats.get('job_cat_Age_mean', row['Age'])
                features.loc[idx, 'exp_vs_job_cat_mean'] = row['Years_of_Experience'] - job_stats.get('job_cat_Years_of_Experience_mean', row['Years_of_Experience'])
                
            else:
                features.loc[idx, 'age_vs_job_cat_mean'] = 0
                features.loc[idx, 'exp_vs_job_cat_mean'] = 0
    
    # Features globales
    if 'global_stats' in stats_dict:
        global_stats = stats_dict['global_stats']
        
        features['age_zscore_global'] = (data['Age'] - global_stats['age_global_mean']) / max(global_stats['age_global_std'], 0.1)
        features['exp_zscore_global'] = (data['Years_of_Experience'] - global_stats['exp_global_mean']) / max(global_stats['exp_global_std'], 0.1)
        
        # Percentiles aproximados (basados en distribución normal)
        from scipy import stats as scipy_stats
        features['age_percentile_global'] = scipy_stats.norm.cdf(features['age_zscore_global'])
        features['exp_percentile_global'] = scipy_stats.norm.cdf(features['exp_zscore_global'])
        
        # Features de texto si están disponibles
        if 'Description' in data.columns:
            features['description_length_zscore'] = (data['Description'].str.len() - global_stats['description_length_mean']) / max(global_stats['description_length_std'], 0.1)
            features['description_word_count_zscore'] = (data['Description'].str.split().str.len() - global_stats['description_word_count_mean']) / max(global_stats['description_word_count_std'], 0.1)
        else:
            features['description_length_zscore'] = 0
            features['description_word_count_zscore'] = 0
    
    # ============= FEATURES ADICIONALES INTELIGENTES =============
    
    # Rankings dentro de grupos (solo si tenemos suficientes datos)
    if len(data) > 10:
        
        # Ranking de edad dentro del grupo educativo
        if 'Education_Level' in data.columns:
            features['age_rank_in_edu_group'] = data.groupby('Education_Level')['Age'].rank(pct=True)
            features['exp_rank_in_edu_group'] = data.groupby('Education_Level')['Years_of_Experience'].rank(pct=True)
        
        # Ranking dentro del grupo de género
        if 'Gender' in data.columns:
            features['age_rank_in_gender'] = data.groupby('Gender')['Age'].rank(pct=True)
            features['exp_rank_in_gender'] = data.groupby('Gender')['Years_of_Experience'].rank(pct=True)
        
        # Ranking dentro de combinaciones
        if 'Education_Level' in data.columns and 'Gender' in data.columns:
            features['age_rank_in_edu_gender'] = data.groupby(['Education_Level', 'Gender'])['Age'].rank(pct=True)
            features['exp_rank_in_edu_gender'] = data.groupby(['Education_Level', 'Gender'])['Years_of_Experience'].rank(pct=True)
    
    # Features de comparación cruzada
    if len(features.columns) > 0:
        # ¿Qué tan consistente es el perfil?
        age_features = [col for col in features.columns if 'age_vs_' in col]
        exp_features = [col for col in features.columns if 'exp_vs_' in col]
        
        if age_features:
            features['age_consistency_score'] = features[age_features].std(axis=1)  # Baja varianza = más consistente
        
        if exp_features:
            features['exp_consistency_score'] = features[exp_features].std(axis=1)
    
    # Limpiar NaN
    features = features.fillna(0)
    
    print(f"   ✅ Creadas {len(features.columns)} features estadísticos para producción")
    
    if is_training:
        return features, stats_dict
    else:
        return features



def create_features_with_stats(data, all_job_categories=None, all_seniority_levels=None, 
                                              stats_dict=None, is_training=True):
    """
    Función integrada que combina features originales + estadísticos seguros para producción
    
    IMPORTANTE: data NO debe contener la columna 'Salary'
    """
    print("🔧 Creando características completas para producción (originales + estadísticos)...")
    
    # Verificar que no hay Salary
    if 'Salary' in data.columns:
        raise ValueError("❌ ERROR: 'Salary' encontrado en data. Usar data sin columna target.")
    
    # 1. Crear features originales
    original_features, feature_names = create_features(data, all_job_categories, all_seniority_levels)
    
    # 2. Crear features estadísticos seguros para producción
    if is_training:
        stat_features, new_stats_dict = create_statistical_features(data, stats_dict, is_training)
        stats_dict = new_stats_dict
    else:
        stat_features = create_statistical_features(data, stats_dict, is_training)
    
    # 3. Combinar
    
    combined_features = pd.concat([original_features, stat_features], axis=1)
    combined_feature_names = list(combined_features.columns)
    
    print(f"✅ Features totales para producción: {len(combined_feature_names)}")
    print(f"   - Originales: {len(feature_names)}")
    print(f"   - Estadísticos: {len(stat_features.columns)}")
    print(f"   ✅ Todas las features son seguras para producción (sin target leakage)")
    
    if is_training:
        return combined_features, combined_feature_names, stats_dict
    else:
        return combined_features, combined_feature_names










def get_enhanced_models(random_state=42):
    """
    Diccionario de modelos mejorado con LightGBM y SVR
    """
    
    models = {
        'Linear Regression': LinearRegression(),
        
        'Ridge Regression': Ridge(
            alpha=1.0, 
            random_state=random_state
        ),
        
        'Random Forest': RandomForestRegressor(
            n_estimators=100, 
            random_state=random_state,
            n_jobs=-1
        ),
        
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            random_state=random_state
        ),
        
        # NUEVOS MODELOS AGREGADOS
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=100,
            random_state=random_state,
            verbose=-1,  # Silenciar output
            n_jobs=-1,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8
        ),
        
        'SVR (RBF)': SVR(
            kernel='rbf',
            C=100,
            gamma='scale',
            epsilon=0.1
        ),
        
        'SVR (Linear)': SVR(
            kernel='linear',
            C=100,
            epsilon=0.1
        )
    }
    
    return models



def analyze_predictions(results):
    """Analizar calidad de predicciones"""
    if not results:
        print("❌ No hay resultados para analizar")
        return None
    
    print("\n📈 Analizando predicciones...")
    
    best_name = results['best_model_name']
    best_result = results['model_results'][best_name]
    
    y_test = results['y_test']
    y_pred = best_result['predictions']
    
    # Crear DataFrame de análisis
    analysis_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'error': y_test - y_pred,
        'abs_error': np.abs(y_test - y_pred),
        'pct_error': np.abs(y_test - y_pred) / y_test * 100
    })
    
    # Estadísticas
    print(f"\nEstadísticas de Error ({best_name}):")
    print(f"   Error promedio: ${analysis_df['error'].mean():,.2f}")
    print(f"   Error absoluto promedio: ${analysis_df['abs_error'].mean():,.2f}")
    print(f"   Error porcentual promedio: {analysis_df['pct_error'].mean():.1f}%")
    print(f"   Predicciones dentro del ±10%: {(analysis_df['pct_error'] <= 10).mean()*100:.1f}%")
    print(f"   Predicciones dentro del ±20%: {(analysis_df['pct_error'] <= 20).mean()*100:.1f}%")
    
    # Visualizaciones
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Actual vs Predicted
    axes[0,0].scatter(analysis_df['actual'], analysis_df['predicted'], alpha=0.6)
    min_val = min(analysis_df['actual'].min(), analysis_df['predicted'].min())
    max_val = max(analysis_df['actual'].max(), analysis_df['predicted'].max())
    axes[0,0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0,0].set_xlabel('Salario Real')
    axes[0,0].set_ylabel('Salario Predicho')
    axes[0,0].set_title('Real vs Predicho')
    
    # 2. Distribución de errores
    axes[0,1].hist(analysis_df['error'], bins=20, alpha=0.7, edgecolor='black')
    axes[0,1].axvline(x=0, color='red', linestyle='--')
    axes[0,1].set_xlabel('Error (Real - Predicho)')
    axes[0,1].set_ylabel('Frecuencia')
    axes[0,1].set_title('Distribución de Errores')
    
    # 3. Error absoluto vs Salario real
    axes[1,0].scatter(analysis_df['actual'], analysis_df['abs_error'], alpha=0.6)
    axes[1,0].set_xlabel('Salario Real')
    axes[1,0].set_ylabel('Error Absoluto')
    axes[1,0].set_title('Error Absoluto vs Salario')
    
    # 4. Error porcentual
    axes[1,1].hist(analysis_df['pct_error'], bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1,1].set_xlabel('Error Porcentual (%)')
    axes[1,1].set_ylabel('Frecuencia')
    axes[1,1].set_title('Error Porcentual')
    
    plt.tight_layout()
    plt.show()
    
    return analysis_df

def analyze_feature_importance(X,feature_names,model):
    """Analizar importancia de características"""
    if not hasattr(model, 'feature_importances_'):
        print("⚠️  El modelo no tiene feature_importances_")
        return None
    
    print("\n🎯 Analizando importancia de características...")
    
    # Crear DataFrame de importancia
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTOP 20 CARACTERÍSTICAS MÁS IMPORTANTES:")
    for i, row in importance_df.head(20).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Visualización
    plt.figure(figsize=(12, 8))
    top_15 = importance_df.head(15)
    
    plt.barh(range(len(top_15)), top_15['importance'])
    plt.yticks(range(len(top_15)), top_15['feature'])
    plt.xlabel('Importancia')
    plt.title('Top 15 Características Más Importantes')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return importance_df



def create_comparison_chart(results):
    """Crear gráfico comparativo de modelos"""
    if not results:
        return None
    
    print("\n📊 Comparación de modelos...")
    
    model_results = results['model_results']
    
    # Preparar datos para visualización
    models = list(model_results.keys())
    rmse_values = [model_results[m]['rmse'] for m in models]
    r2_values = [model_results[m]['r2'] for m in models]
    cv_rmse_values = [model_results[m]['cv_rmse'] for m in models]
    
    # Crear gráficos
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # RMSE Test
    axes[0].bar(models, rmse_values, alpha=0.7, color='red')
    axes[0].set_title('RMSE en Conjunto de Test')
    axes[0].set_ylabel('RMSE ($)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # R² Test
    axes[1].bar(models, r2_values, alpha=0.7, color='blue')
    axes[1].set_title('R² en Conjunto de Test')
    axes[1].set_ylabel('R²')
    axes[1].tick_params(axis='x', rotation=45)
    
    # CV RMSE
    axes[2].bar(models, cv_rmse_values, alpha=0.7, color='green')
    axes[2].set_title('RMSE Cross-Validation')
    axes[2].set_ylabel('CV RMSE ($)')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    


