import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score




def create_features(data):
    """
    Función unificada que combina todas las features:
    - Features originales existentes
    - Nuevos ratios sugeridos  
    - Agrupaciones inteligentes de job titles
    - Features de texto mejoradas
    """
    print("🔧 Creando todas las características mejoradas...")
    
    features = pd.DataFrame()
    
    # ============= CARACTERÍSTICAS BÁSICAS ORIGINALES =============
    features['age'] = data['Age']
    features['years_experience'] = data['Years_of_Experience']
    features['age_experience_ratio'] = data['Age'] / (data['Years_of_Experience'] + 1)
    features['experience_squared'] = data['Years_of_Experience'] ** 2
    
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
    
    # Crear dummies para categorías funcionales
    category_dummies = pd.get_dummies(job_categories, prefix='job_cat')
    features = pd.concat([features, category_dummies], axis=1)
    
    # Crear dummies para seniority
    seniority_dummies = pd.get_dummies(seniority_levels, prefix='seniority')
    features = pd.concat([features, seniority_dummies], axis=1)
    
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
    
    # Score de demanda de mercado
    demand_scores = {
        'DATA_SCIENCE': 10, 'ENGINEERING': 9, 'EXECUTIVE': 8, 'FINANCE': 7,
        'CONSULTING': 7, 'ANALYST': 6, 'MANAGEMENT': 6, 'SALES': 5,
        'MARKETING': 5, 'DESIGN': 4, 'OPERATIONS': 4, 'OTHER': 2
    }
    #features['job_market_demand_score'] = job_categories.map(demand_scores)
    
    # ============= INTERACCIONES COMPLEJAS =============
    
    # Combos valiosos
    features['tech_senior_combo'] = features['is_tech_role'] * features['is_senior_role']
    features['tech_management_combo'] = features['is_tech_role'] * features['is_management_role']
    #features['male_senior_interaction'] = features['gender_male'] * features['job_senior']
    #features['female_senior_interaction'] = features['gender_female'] * features['job_senior']
    """
    # Score profesional compuesto
    features['professional_value_score'] = (
        features['years_experience'] * 0.3 +
        features['edu_phd'] * 15 +
        features['edu_masters'] * 8 +
        features['job_senior'] * 10 +
        features['is_management_role'] * 20 +
        features['leadership_terms_count'] * 2
    )
    """
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


# modelos a probar

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



def train_models(X, y, test_size=0.2, random_state=42):
    """
    Entrenar y comparar modelos mejorados incluyendo LightGBM y SVR
    """
    print("\n🚀 Entrenando modelos mejorados...")
    
    # División de datos
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import numpy as np
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Crear escalador
    scaler = StandardScaler()
    
    # Escalar características
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Obtener modelos
    models = get_enhanced_models(random_state)
    
    results = {}
    
    # Modelos que necesitan escalado
    scaled_models = ['Linear Regression', 'Ridge Regression', 'SVR (RBF)', 'SVR (Linear)']
    
    for name, model in models.items():
        print(f"   Entrenando {name}...")
        
        try:
            # Seleccionar datos apropiados
            if name in scaled_models:
                X_train_model = X_train_scaled
                X_test_model = X_test_scaled
                X_cv = X_train_scaled
            else:
                X_train_model = X_train
                X_test_model = X_test
                X_cv = X_train
            
            # Entrenar modelo
            model.fit(X_train_model, y_train)
            y_pred = model.predict(X_test_model)
            
            # Métricas
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_cv, y_train, cv=5, 
                scoring='neg_mean_squared_error', n_jobs=-1
            )
            cv_rmse = np.sqrt(-cv_scores.mean())
            cv_std = np.sqrt(cv_scores.std())
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'cv_rmse': cv_rmse,
                'cv_std': cv_std,
                'predictions': y_pred,
                'scaled': name in scaled_models
            }
            
            print(f"      RMSE: ${rmse:,.2f}")
            print(f"      R²: {r2:.3f}")
            print(f"      CV RMSE: ${cv_rmse:,.2f} (±{cv_std:,.2f})")
            
        except Exception as e:
            print(f"      ❌ Error entrenando {name}: {str(e)}")
            continue
    
    # Seleccionar mejor modelo
    valid_results = {k: v for k, v in results.items() if 'cv_rmse' in v}
    
    if valid_results:
        best_model_name = min(valid_results.keys(), key=lambda x: valid_results[x]['cv_rmse'])
        best_model = valid_results[best_model_name]['model']
        
        print(f"\n🏆 Mejor modelo: {best_model_name}")
        print(f"   RMSE: ${valid_results[best_model_name]['rmse']:,.2f}")
        print(f"   R²: {valid_results[best_model_name]['r2']:.3f}")
        print(f"   CV RMSE: ${valid_results[best_model_name]['cv_rmse']:,.2f}")
    else:
        print("❌ No se pudieron entrenar modelos válidos")
        best_model_name = None
        best_model = None
    
    # Resultado final
    final_results = {
        'model_results': results,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'scaler': scaler,
        'X_test': X_test,
        'y_test': y_test,
        'X_train': X_train,
        'y_train': y_train
    }
    
    return final_results,best_model


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
    
 



#Combito de métricas


def comprehensive_model_evaluation(train_results, cv=5):
    """
    Evaluación comprensiva con múltiples métricas para seleccionar el mejor modelo
    
    Parameters:
    -----------
    train_results : dict
        Resultados completos de train_models() que incluye:
        - model_results: dict con resultados de cada modelo
        - X_test, y_test: datos de test
        - X_train, y_train: datos de entrenamiento
    cv : int
        Número de folds para cross-validation
    
    Returns:
    --------
    comprehensive_results : DataFrame
        Tabla comparativa con todas las métricas
    recommendations : dict
        Recomendaciones por caso de uso
    """
    
    print("📊 EVALUACIÓN COMPRENSIVA DE MODELOS")
    print("="*70)
    
    evaluation_results = []
    
    # Extraer datos correctos
    models_dict = train_results['model_results']
    X_test = train_results['X_test']
    y_test = train_results['y_test']
    X_train = train_results['X_train']
    y_train = train_results['y_train']
    
    print(f"📏 Datos: X_test: {X_test.shape}, y_test: {len(y_test)}")
    
    for model_name, model_info in models_dict.items():
        print(f"\n📋 Evaluando {model_name}...")
        
        try:
            model = model_info['model']
            y_pred = model_info['predictions']  # Ya son predicciones en X_test
            
            # Verificar tamaños
            if len(y_pred) != len(y_test):
                print(f"   ⚠️ Ajustando tamaños: y_pred={len(y_pred)}, y_test={len(y_test)}")
                min_len = min(len(y_pred), len(y_test))
                y_pred = y_pred[:min_len]
                y_test_subset = y_test.iloc[:min_len] if hasattr(y_test, 'iloc') else y_test[:min_len]
            else:
                y_test_subset = y_test
            
            # ============= MÉTRICAS DE EXACTITUD =============
            
            # R² (Coeficiente de Determinación)
            r2 = r2_score(y_test_subset, y_pred)
            
            # RMSE (Root Mean Squared Error)
            rmse = np.sqrt(mean_squared_error(y_test_subset, y_pred))
            
            # MAE (Mean Absolute Error)
            mae = mean_absolute_error(y_test_subset, y_pred)
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_test_subset - y_pred) / y_test_subset)) * 100
            
            # ============= MÉTRICAS DE ROBUSTEZ =============
            
            # Usar X_train y y_train para cross-validation
            # Determinar si el modelo necesita escalado
            needs_scaling = model_name in ['Linear Regression', 'Ridge Regression', 'SVR (RBF)', 'SVR (Linear)']
            
            if needs_scaling and 'scaler' in train_results:
                scaler = train_results['scaler']
                X_for_cv = scaler.transform(X_train)
            else:
                X_for_cv = X_train
            
            # Cross-validation R²
            try:
                cv_r2_scores = cross_val_score(model, X_for_cv, y_train, cv=cv, scoring='r2', n_jobs=-1)
                cv_r2_mean = cv_r2_scores.mean()
                cv_r2_std = cv_r2_scores.std()
            except Exception as e:
                print(f"      ⚠️ CV R² falló: {str(e)}")
                cv_r2_mean = r2
                cv_r2_std = 0
            
            # Cross-validation RMSE
            try:
                cv_rmse_scores = cross_val_score(model, X_for_cv, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
                cv_rmse_mean = np.sqrt(-cv_rmse_scores.mean())
                cv_rmse_std = np.sqrt(cv_rmse_scores.std())
            except Exception as e:
                print(f"      ⚠️ CV RMSE falló: {str(e)}")
                cv_rmse_mean = rmse
                cv_rmse_std = 0
            
            # ============= MÉTRICAS DE ESTABILIDAD =============
            
            # Varianza de las predicciones (diversidad)
            pred_variance = np.var(y_pred)
            
            # Rango de errores
            errors = y_test_subset - y_pred
            error_range = np.max(errors) - np.min(errors)
            
            # Percentil 95 de errores absolutos (outliers)
            abs_errors = np.abs(errors)
            error_p95 = np.percentile(abs_errors, 95)
            
            # ============= MÉTRICAS DE NEGOCIO =============
            
            # Error promedio en salario
            avg_salary_error = mae
            
            # Porcentaje de predicciones dentro del ±10%
            within_10pct = np.mean(np.abs(errors / y_test_subset) <= 0.10) * 100
            
            # Porcentaje de predicciones dentro del ±20%
            within_20pct = np.mean(np.abs(errors / y_test_subset) <= 0.20) * 100
            
            # ============= SCORES COMPUESTOS =============
            
            # Score de Precisión (combina R² y RMSE normalizado)
            rmse_normalized = rmse / np.mean(y_test_subset)
            precision_score = r2 * (1 - min(rmse_normalized, 1))  # Evitar valores negativos
            
            # Score de Robustez (estabilidad en CV)
            robustness_score = cv_r2_mean * (1 - cv_r2_std)
            
            # Score de Negocio (combina precisión y aplicabilidad práctica)
            business_score = (r2 * 0.4) + ((within_20pct/100) * 0.4) + ((1 - min(mape/100, 1)) * 0.2)
            
            # ============= GUARDAR RESULTADOS =============
            
            evaluation_results.append({
                'Model': model_name,
                
                # Métricas básicas
                'R²': r2,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                
                # Cross-validation
                'CV_R²_mean': cv_r2_mean,
                'CV_R²_std': cv_r2_std,
                'CV_RMSE_mean': cv_rmse_mean,
                'CV_RMSE_std': cv_rmse_std,
                
                # Estabilidad
                'Pred_Variance': pred_variance,
                'Error_Range': error_range,
                'Error_P95': error_p95,
                
                # Negocio
                'Within_10%': within_10pct,
                'Within_20%': within_20pct,
                
                # Scores compuestos
                'Precision_Score': precision_score,
                'Robustness_Score': robustness_score,
                'Business_Score': business_score
            })
            
        except Exception as e:
            print(f"   ❌ Error evaluando {model_name}: {str(e)}")
            continue
    
    # Crear DataFrame de resultados
    results_df = pd.DataFrame(evaluation_results)
    
    if len(results_df) == 0:
        print("❌ No se pudieron evaluar modelos")
        return None, None
    
    # ============= ANÁLISIS Y RECOMENDACIONES =============
    
    recommendations = analyze_and_recommend(results_df)
    
    return results_df, recommendations

def analyze_and_recommend(results_df):
    """
    Analizar resultados y dar recomendaciones por caso de uso
    """
    
    recommendations = {}
    
    # ============= CASO 1: MÁXIMA PRECISIÓN =============
    best_r2_idx = results_df['R²'].idxmax()
    recommendations['max_precision'] = {
        'model': results_df.loc[best_r2_idx, 'Model'],
        'reason': 'Máximo R² (mejor explicación de varianza)',
        'metrics': {
            'R²': results_df.loc[best_r2_idx, 'R²'],
            'RMSE': results_df.loc[best_r2_idx, 'RMSE'],
            'MAE': results_df.loc[best_r2_idx, 'MAE']
        }
    }
    
    # ============= CASO 2: MÍNIMO ERROR ABSOLUTO =============
    best_rmse_idx = results_df['RMSE'].idxmin()
    recommendations['min_error'] = {
        'model': results_df.loc[best_rmse_idx, 'Model'],
        'reason': 'Mínimo RMSE (menor error cuadrático)',
        'metrics': {
            'R²': results_df.loc[best_rmse_idx, 'R²'],
            'RMSE': results_df.loc[best_rmse_idx, 'RMSE'],
            'MAE': results_df.loc[best_rmse_idx, 'MAE']
        }
    }
    
    # ============= CASO 3: MÁXIMA ROBUSTEZ =============
    best_robust_idx = results_df['Robustness_Score'].idxmax()
    recommendations['max_robustness'] = {
        'model': results_df.loc[best_robust_idx, 'Model'],
        'reason': 'Máxima robustez (estable en cross-validation)',
        'metrics': {
            'CV_R²_mean': results_df.loc[best_robust_idx, 'CV_R²_mean'],
            'CV_R²_std': results_df.loc[best_robust_idx, 'CV_R²_std'],
            'CV_RMSE_mean': results_df.loc[best_robust_idx, 'CV_RMSE_mean']
        }
    }
    
    # ============= CASO 4: MEJOR PARA NEGOCIO =============
    best_business_idx = results_df['Business_Score'].idxmax()
    recommendations['best_business'] = {
        'model': results_df.loc[best_business_idx, 'Model'],
        'reason': 'Mejor balance precision/aplicabilidad práctica',
        'metrics': {
            'Within_20%': results_df.loc[best_business_idx, 'Within_20%'],
            'MAPE': results_df.loc[best_business_idx, 'MAPE'],
            'Business_Score': results_df.loc[best_business_idx, 'Business_Score']
        }
    }
    
    # ============= CASO 5: BALANCE GENERAL =============
    # Score compuesto: 40% precisión + 30% robustez + 30% negocio
    results_df['Overall_Score'] = (
        (results_df['R²'] * 0.4) + 
        (results_df['Robustness_Score'] * 0.3) + 
        (results_df['Business_Score'] * 0.3)
    )
    
    best_overall_idx = results_df['Overall_Score'].idxmax()
    recommendations['best_overall'] = {
        'model': results_df.loc[best_overall_idx, 'Model'],
        'reason': 'Mejor balance general (precisión + robustez + negocio)',
        'metrics': {
            'R²': results_df.loc[best_overall_idx, 'R²'],
            'CV_R²_mean': results_df.loc[best_overall_idx, 'CV_R²_mean'],
            'Within_20%': results_df.loc[best_overall_idx, 'Within_20%'],
            'Overall_Score': results_df.loc[best_overall_idx, 'Overall_Score']
        }
    }
    
    return recommendations

def print_comprehensive_analysis(results_df, recommendations):
    """
    Imprimir análisis comprensivo y recomendaciones
    """
    
    print("\n📊 TABLA COMPARATIVA COMPLETA:")
    print("="*100)
    
    # Tabla principal con métricas clave
    key_metrics = ['Model', 'R²', 'RMSE', 'MAE', 'CV_R²_mean', 'CV_RMSE_mean', 'Within_20%', 'Business_Score']
    display_df = results_df[key_metrics].round(3)
    print(display_df.to_string(index=False))
    
    print(f"\n🎯 RECOMENDACIONES POR CASO DE USO:")
    print("="*70)
    
    use_cases = {
        'max_precision': '🎯 MÁXIMA PRECISIÓN (Investigación/Academia)',
        'min_error': '📉 MÍNIMO ERROR (Aplicaciones críticas)',
        'max_robustness': '🛡️  MÁXIMA ROBUSTEZ (Producción estable)',
        'best_business': '💼 MEJOR PARA NEGOCIO (Decisiones prácticas)',
        'best_overall': '⭐ BALANCE GENERAL (Recomendado)'
    }
    
    for case, description in use_cases.items():
        if case in recommendations:
            rec = recommendations[case]
            print(f"\n{description}")
            print(f"   🏆 Modelo recomendado: {rec['model']}")
            print(f"   📋 Razón: {rec['reason']}")
            
            # Mostrar métricas clave
            for metric, value in rec['metrics'].items():
                if isinstance(value, float):
                    if 'RMSE' in metric or 'MAE' in metric:
                        print(f"   📊 {metric}: ${value:,.0f}")
                    else:
                        print(f"   📊 {metric}: {value:.3f}")
                else:
                    print(f"   📊 {metric}: {value}")

