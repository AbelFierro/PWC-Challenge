import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, cross_validate, KFold
from scipy import stats
import seaborn as sns
from scipy.stats import t, norm
import warnings
warnings.filterwarnings('ignore')


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
        
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=100,
            random_state=random_state,
            verbose=-1,
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


def calculate_confidence_intervals(scores, confidence_level=0.95):
    """
    Calcular intervalos de confianza para métricas de cross-validation
    
    Parameters:
    -----------
    scores : array-like
        Scores de cross-validation
    confidence_level : float
        Nivel de confianza (por defecto 95%)
    
    Returns:
    --------
    dict con estadísticas de intervalo de confianza
    """
    
    n = len(scores)
    mean_score = np.mean(scores)
    std_score = np.std(scores, ddof=1)  # Usar desviación estándar muestral
    se_score = std_score / np.sqrt(n)  # Error estándar
    
    # Grados de libertad
    df = n - 1
    
    # Valor crítico de la distribución t
    alpha = 1 - confidence_level
    t_critical = t.ppf(1 - alpha/2, df)
    
    # Intervalo de confianza
    margin_error = t_critical * se_score
    ci_lower = mean_score - margin_error
    ci_upper = mean_score + margin_error
    
    return {
        'mean': mean_score,
        'std': std_score,
        'se': se_score,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'margin_error': margin_error,
        'confidence_level': confidence_level,
        'n_samples': n
    }


def bootstrap_confidence_interval(scores, confidence_level=0.95, n_bootstrap=1000):
    """
    Calcular intervalo de confianza usando bootstrap
    
    Parameters:
    -----------
    scores : array-like
        Scores originales
    confidence_level : float
        Nivel de confianza
    n_bootstrap : int
        Número de muestras bootstrap
    
    Returns:
    --------
    dict con estadísticas bootstrap
    """
    
    bootstrap_means = []
    n = len(scores)
    
    # Generar muestras bootstrap
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(scores, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Calcular percentiles para intervalo de confianza
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_means, (alpha/2) * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    return {
        'mean': np.mean(bootstrap_means),
        'std': np.std(bootstrap_means),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence_level,
        'bootstrap_means': bootstrap_means
    }


def train_models_with_confidence_intervals(X, y, test_size=0.2, random_state=42, cv_folds=10, confidence_level=0.95):
    """
    Entrenar modelos con intervalos de confianza estadísticos
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target variable
    test_size : float
        Proporción para test set
    random_state : int
        Semilla aleatoria
    cv_folds : int
        Número de folds para cross-validation
    confidence_level : float
        Nivel de confianza para intervalos
    
    Returns:
    --------
    dict con resultados completos incluyendo intervalos de confianza
    """
    
    print(f"\n🚀 Entrenando modelos con intervalos de confianza al {confidence_level*100:.0f}%...")
    print(f"📊 Cross-validation: {cv_folds} folds")
    
    # División de datos
    from sklearn.model_selection import train_test_split, cross_validate
    from sklearn.preprocessing import StandardScaler
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    # Crear escalador
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Configurar cross-validation
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Métricas a evaluar
    scoring_metrics = {
        'r2': 'r2',
        'neg_mse': 'neg_mean_squared_error',
        'neg_mae': 'neg_mean_absolute_error'
    }
    
    # Obtener modelos
    models = get_enhanced_models(random_state)
    
    results = {}
    scaled_models = ['Linear Regression', 'Ridge Regression', 'SVR (RBF)', 'SVR (Linear)']
    
    for model_name, model in models.items():
        print(f"\n   📈 Evaluando {model_name}...")
        
        try:
            # Seleccionar datos apropiados
            if model_name in scaled_models:
                X_train_model = X_train_scaled
                X_test_model = X_test_scaled
                X_cv = X_train_scaled
            else:
                X_train_model = X_train
                X_test_model = X_test
                X_cv = X_train
            
            # ============= ENTRENAMIENTO Y PREDICCIÓN =============
            
            # Entrenar modelo
            model.fit(X_train_model, y_train)
            y_pred = model.predict(X_test_model)
            
            # Métricas en test set
            test_r2 = r2_score(y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_mae = mean_absolute_error(y_test, y_pred)
            test_mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            # ============= CROSS-VALIDATION CON INTERVALOS DE CONFIANZA =============
            
            cv_results = cross_validate(
                model, X_cv, y_train, 
                cv=cv, 
                scoring=scoring_metrics, 
                n_jobs=-1,
                return_train_score=True
            )
            
            # Procesar resultados de CV
            cv_r2_scores = cv_results['test_r2']
            cv_rmse_scores = np.sqrt(-cv_results['test_neg_mse'])
            cv_mae_scores = -cv_results['test_neg_mae']
            
            # Calcular intervalos de confianza
            r2_ci = calculate_confidence_intervals(cv_r2_scores, confidence_level)
            rmse_ci = calculate_confidence_intervals(cv_rmse_scores, confidence_level)
            mae_ci = calculate_confidence_intervals(cv_mae_scores, confidence_level)
            
            # Bootstrap para R² (método alternativo)
            r2_bootstrap = bootstrap_confidence_interval(cv_r2_scores, confidence_level)
            
            # ============= MÉTRICAS ADICIONALES =============
            
            # Error porcentual
            pct_errors = np.abs((y_test - y_pred) / y_test) * 100
            within_10pct = np.mean(pct_errors <= 10) * 100
            within_20pct = np.mean(pct_errors <= 20) * 100
            
            # Estabilidad del modelo
            cv_stability = 1 - (r2_ci['std'] / max(abs(r2_ci['mean']), 0.001))
            
            # Score de confianza (basado en ancho del intervalo)
            confidence_score = 1 - (r2_ci['margin_error'] / max(abs(r2_ci['mean']), 0.001))
            
            # ============= GUARDAR RESULTADOS =============
            
            results[model_name] = {
                'model': model,
                'scaled': model_name in scaled_models,
                
                # Métricas de test
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_mape': test_mape,
                'predictions': y_pred,
                
                # Cross-validation con intervalos de confianza
                'cv_r2': r2_ci,
                'cv_rmse': rmse_ci,
                'cv_mae': mae_ci,
                
                # Bootstrap
                'r2_bootstrap': r2_bootstrap,
                
                # Métricas de negocio
                'within_10pct': within_10pct,
                'within_20pct': within_20pct,
                
                # Métricas de estabilidad
                'cv_stability': cv_stability,
                'confidence_score': confidence_score,
                
                # Datos raw de CV
                'cv_r2_scores': cv_r2_scores,
                'cv_rmse_scores': cv_rmse_scores,
                'cv_mae_scores': cv_mae_scores
            }
            
            print(f"      📊 Test R²: {test_r2:.3f}")
            print(f"      📊 Test RMSE: ${test_rmse:,.0f}")
            print(f"      📊 CV R² (IC {confidence_level*100:.0f}%): {r2_ci['mean']:.3f} [{r2_ci['ci_lower']:.3f}, {r2_ci['ci_upper']:.3f}]")
            print(f"      📊 CV RMSE (IC {confidence_level*100:.0f}%): ${rmse_ci['mean']:,.0f} [${rmse_ci['ci_lower']:,.0f}, ${rmse_ci['ci_upper']:,.0f}]")
            print(f"      📊 Estabilidad: {cv_stability:.3f}")
            
        except Exception as e:
            print(f"      ❌ Error entrenando {model_name}: {str(e)}")
            continue
    
    # ============= SELECCIONAR MEJOR MODELO =============
    
    valid_results = {k: v for k, v in results.items() if 'cv_r2' in v}
    
    if valid_results:
        # Criterio: mayor R² medio con menor incertidumbre
        best_model_name = max(
            valid_results.keys(), 
            key=lambda x: valid_results[x]['cv_r2']['mean'] * valid_results[x]['confidence_score']
        )
        best_model = valid_results[best_model_name]['model']
        
        print(f"\n🏆 Mejor modelo (R² × Confianza): {best_model_name}")
        print(f"   📊 CV R²: {valid_results[best_model_name]['cv_r2']['mean']:.3f}")
        print(f"   📊 Confidence Score: {valid_results[best_model_name]['confidence_score']:.3f}")
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
        'y_train': y_train,
        'confidence_level': confidence_level,
        'cv_folds': cv_folds
    }
    
    return final_results, best_model


def create_confidence_interval_visualization(results, figsize=(16, 12)):
    """
    Crear visualizaciones comprehensivas de intervalos de confianza
    
    Parameters:
    -----------
    results : dict
        Resultados de train_models_with_confidence_intervals
    figsize : tuple
        Tamaño de la figura
    """
    
    if not results or 'model_results' not in results:
        print("❌ No hay resultados para visualizar")
        return
    
    model_results = results['model_results']
    confidence_level = results.get('confidence_level', 0.95)
    
    # Preparar datos para visualización
    models = list(model_results.keys())
    r2_means = [model_results[m]['cv_r2']['mean'] for m in models]
    r2_cis = [(model_results[m]['cv_r2']['ci_lower'], model_results[m]['cv_r2']['ci_upper']) for m in models]
    rmse_means = [model_results[m]['cv_rmse']['mean'] for m in models]
    rmse_cis = [(model_results[m]['cv_rmse']['ci_lower'], model_results[m]['cv_rmse']['ci_upper']) for m in models]
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'Análisis de Modelos con Intervalos de Confianza ({confidence_level*100:.0f}%)', 
                 fontsize=16, fontweight='bold')
    
    # ============= 1. R² CON INTERVALOS DE CONFIANZA =============
    
    ax1 = axes[0, 0]
    y_positions = np.arange(len(models))
    
    # Barras con intervalos de confianza
    bars = ax1.barh(y_positions, r2_means, alpha=0.7, color='skyblue')
    
    # Intervalos de confianza
    for i, (lower, upper) in enumerate(r2_cis):
        ax1.plot([lower, upper], [i, i], color='red', linewidth=2, marker='|', markersize=8)
    
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(models)
    ax1.set_xlabel('R² Score')
    ax1.set_title('R² con Intervalos de Confianza')
    ax1.grid(axis='x', alpha=0.3)
    
    # ============= 2. RMSE CON INTERVALOS DE CONFIANZA =============
    
    ax2 = axes[0, 1]
    
    # Barras con intervalos de confianza
    bars = ax2.barh(y_positions, rmse_means, alpha=0.7, color='lightcoral')
    
    # Intervalos de confianza
    for i, (lower, upper) in enumerate(rmse_cis):
        ax2.plot([lower, upper], [i, i], color='darkred', linewidth=2, marker='|', markersize=8)
    
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(models)
    ax2.set_xlabel('RMSE ($)')
    ax2.set_title('RMSE con Intervalos de Confianza')
    ax2.grid(axis='x', alpha=0.3)
    
    # ============= 3. DISTRIBUCIÓN DE R² POR MODELO =============
    
    ax3 = axes[0, 2]
    
    # Box plot de distribuciones de R²
    r2_distributions = [model_results[m]['cv_r2_scores'] for m in models]
    box_plot = ax3.boxplot(r2_distributions, labels=[m.replace(' ', '\n') for m in models], 
                          patch_artist=True, showmeans=True)
    
    # Colorear cajas
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('R² Score')
    ax3.set_title('Distribución de R² (Cross-Validation)')
    ax3.grid(axis='y', alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # ============= 4. ESTABILIDAD VS RENDIMIENTO =============
    
    ax4 = axes[1, 0]
    
    stability_scores = [model_results[m]['cv_stability'] for m in models]
    confidence_scores = [model_results[m]['confidence_score'] for m in models]
    
    scatter = ax4.scatter(stability_scores, r2_means, 
                         s=100, alpha=0.7, c=confidence_scores, 
                         cmap='viridis', edgecolors='black')
    
    # Añadir etiquetas
    for i, model in enumerate(models):
        ax4.annotate(model, (stability_scores[i], r2_means[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Estabilidad del Modelo')
    ax4.set_ylabel('R² Promedio')
    ax4.set_title('Estabilidad vs Rendimiento')
    ax4.grid(alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Score de Confianza')
    
    # ============= 5. COMPARACIÓN DE MÉTODOS DE IC =============
    
    ax5 = axes[1, 1]
    
    # Comparar intervalos t-student vs bootstrap para R²
    model_sample = models[0]  # Usar primer modelo como ejemplo
    
    t_ci = model_results[model_sample]['cv_r2']
    bootstrap_ci = model_results[model_sample]['r2_bootstrap']
    
    methods = ['T-Student', 'Bootstrap']
    means = [t_ci['mean'], bootstrap_ci['mean']]
    cis = [(t_ci['ci_lower'], t_ci['ci_upper']), 
           (bootstrap_ci['ci_lower'], bootstrap_ci['ci_upper'])]
    
    y_pos = [0, 1]
    bars = ax5.barh(y_pos, means, alpha=0.7, color=['lightblue', 'lightgreen'])
    
    # Intervalos de confianza
    for i, (lower, upper) in enumerate(cis):
        ax5.plot([lower, upper], [i, i], color='red', linewidth=3, marker='|', markersize=10)
    
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(methods)
    ax5.set_xlabel('R² Score')
    ax5.set_title(f'Comparación de Métodos IC\n({model_sample})')
    ax5.grid(axis='x', alpha=0.3)
    
    # ============= 6. ANCHO DE INTERVALOS DE CONFIANZA =============
    
    ax6 = axes[1, 2]
    
    # Calcular anchos de intervalos
    r2_widths = [model_results[m]['cv_r2']['ci_upper'] - model_results[m]['cv_r2']['ci_lower'] 
                 for m in models]
    rmse_widths = [model_results[m]['cv_rmse']['ci_upper'] - model_results[m]['cv_rmse']['ci_lower'] 
                   for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, r2_widths, width, label='R² IC Width', alpha=0.7, color='lightblue')
    
    # Normalizar RMSE widths para segunda escala
    rmse_widths_norm = np.array(rmse_widths) / max(rmse_widths) * max(r2_widths)
    bars2 = ax6.bar(x + width/2, rmse_widths_norm, width, label='RMSE IC Width (norm)', alpha=0.7, color='lightcoral')
    
    ax6.set_xlabel('Modelos')
    ax6.set_ylabel('Ancho del Intervalo')
    ax6.set_title('Precisión de Estimaciones\n(Menor ancho = Mayor precisión)')
    ax6.set_xticks(x)
    ax6.set_xticklabels([m.replace(' ', '\n') for m in models], rotation=45, ha='right')
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def create_statistical_summary_table(results):
    """
    Crear tabla resumen con estadísticas completas
    """
    
    if not results or 'model_results' not in results:
        print("❌ No hay resultados para resumir")
        return None
    
    model_results = results['model_results']
    confidence_level = results.get('confidence_level', 0.95)
    
    print(f"\n📊 RESUMEN ESTADÍSTICO COMPLETO (IC {confidence_level*100:.0f}%)")
    print("="*120)
    
    summary_data = []
    
    for model_name, model_info in model_results.items():
        
        r2_ci = model_info['cv_r2']
        rmse_ci = model_info['cv_rmse']
        mae_ci = model_info['cv_mae']
        
        summary_data.append({
            'Modelo': model_name,
            'Test_R²': model_info['test_r2'],
            'CV_R²_Mean': r2_ci['mean'],
            'CV_R²_IC_Lower': r2_ci['ci_lower'],
            'CV_R²_IC_Upper': r2_ci['ci_upper'],
            'CV_R²_MarginError': r2_ci['margin_error'],
            'CV_RMSE_Mean': rmse_ci['mean'],
            'CV_RMSE_IC_Lower': rmse_ci['ci_lower'],
            'CV_RMSE_IC_Upper': rmse_ci['ci_upper'],
            'CV_MAE_Mean': mae_ci['mean'],
            'Within_20%': model_info['within_20pct'],
            'Estabilidad': model_info['cv_stability'],
            'Confidence_Score': model_info['confidence_score']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Formatear para display
    display_df = summary_df.copy()
    
    # Redondear valores numéricos
    numeric_cols = ['Test_R²', 'CV_R²_Mean', 'CV_R²_IC_Lower', 'CV_R²_IC_Upper', 
                   'CV_R²_MarginError', 'CV_RMSE_Mean', 'CV_RMSE_IC_Lower', 
                   'CV_RMSE_IC_Upper', 'CV_MAE_Mean', 'Within_20%', 'Estabilidad', 'Confidence_Score']
    
    for col in numeric_cols:
        if 'RMSE' in col or 'MAE' in col:
            display_df[col] = display_df[col].round(0).astype(int)
        else:
            display_df[col] = display_df[col].round(3)
    
    print(display_df.to_string(index=False))
    
    return summary_df


def statistical_model_comparison(results, alpha=0.05):
    """
    Realizar comparaciones estadísticas entre modelos usando tests de hipótesis
    
    Parameters:
    -----------
    results : dict
        Resultados de train_models_with_confidence_intervals
    alpha : float
        Nivel de significancia para tests estadísticos
    
    Returns:
    --------
    dict con resultados de comparaciones estadísticas
    """
    
    print(f"\n🔬 COMPARACIÓN ESTADÍSTICA ENTRE MODELOS (α = {alpha})")
    print("="*80)
    
    model_results = results['model_results']
    models = list(model_results.keys())
    
    # Extraer scores de CV para cada modelo
    cv_scores = {}
    for model_name in models:
        cv_scores[model_name] = model_results[model_name]['cv_r2_scores']
    
    comparison_results = {
        'pairwise_tests': {},
        'rankings': {},
        'significant_differences': []
    }
    
    # ============= TESTS PAREADOS =============
    
    print("\n📈 Tests t pareados (R² Cross-Validation):")
    print("-" * 60)
    
    from scipy.stats import ttest_rel
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models[i+1:], i+1):
            
            scores1 = cv_scores[model1]
            scores2 = cv_scores[model2]
            
            # Test t pareado
            statistic, p_value = ttest_rel(scores1, scores2)
            
            # Determinar significancia
            significant = p_value < alpha
            
            # Calcular diferencia promedio
            mean_diff = np.mean(scores1) - np.mean(scores2)
            
            comparison_results['pairwise_tests'][f"{model1} vs {model2}"] = {
                'statistic': statistic,
                'p_value': p_value,
                'significant': significant,
                'mean_difference': mean_diff,
                'better_model': model1 if mean_diff > 0 else model2
            }
            
            # Mostrar resultado
            significance_indicator = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if significant else ""
            better_model = model1 if mean_diff > 0 else model2
            
            print(f"{model1:15} vs {model2:15} | "
                  f"Diff: {mean_diff:+.4f} | "
                  f"p-value: {p_value:.4f} {significance_indicator} | "
                  f"Mejor: {better_model if significant else 'No diferencia'}")
            
            if significant:
                comparison_results['significant_differences'].append({
                    'comparison': f"{model1} vs {model2}",
                    'better_model': better_model,
                    'p_value': p_value,
                    'effect_size': abs(mean_diff)
                })
    
    # ============= RANKING ESTADÍSTICO =============
    
    print(f"\n🏆 RANKING DE MODELOS (basado en R² CV):")
    print("-" * 50)
    
    # Calcular estadísticas para ranking
    ranking_data = []
    for model_name in models:
        scores = cv_scores[model_name]
        r2_ci = model_results[model_name]['cv_r2']
        
        ranking_data.append({
            'model': model_name,
            'mean_r2': np.mean(scores),
            'ci_lower': r2_ci['ci_lower'],
            'ci_upper': r2_ci['ci_upper'],
            'stability': model_results[model_name]['cv_stability'],
            'confidence_score': model_results[model_name]['confidence_score']
        })
    
    # Ordenar por R² promedio
    ranking_data.sort(key=lambda x: x['mean_r2'], reverse=True)
    
    for i, data in enumerate(ranking_data, 1):
        ci_width = data['ci_upper'] - data['ci_lower']
        
        print(f"{i:2d}. {data['model']:18} | "
              f"R²: {data['mean_r2']:.4f} | "
              f"IC: [{data['ci_lower']:.3f}, {data['ci_upper']:.3f}] | "
              f"Estab: {data['stability']:.3f}")
    
    comparison_results['rankings']['by_mean_r2'] = ranking_data
    
    # ============= ANÁLISIS DE SIGNIFICANCIA =============
    
    if comparison_results['significant_differences']:
        print(f"\n🔍 DIFERENCIAS ESTADÍSTICAMENTE SIGNIFICATIVAS:")
        print("-" * 60)
        
        sig_diffs = sorted(comparison_results['significant_differences'], 
                          key=lambda x: x['p_value'])
        
        for diff in sig_diffs:
            print(f"• {diff['comparison']}")
            print(f"  Mejor modelo: {diff['better_model']}")
            print(f"  p-value: {diff['p_value']:.6f}")
            print(f"  Tamaño del efecto: {diff['effect_size']:.4f}")
            print()
    else:
        print(f"\n⚠️  No se encontraron diferencias estadísticamente significativas entre modelos")
        print(f"   (con α = {alpha})")
    
    return comparison_results


def recommend_best_model_statistical(results, use_case='general'):
    """
    Recomendar el mejor modelo basado en análisis estadístico
    
    Parameters:
    -----------
    results : dict
        Resultados completos del análisis
    use_case : str
        Caso de uso específico: 'general', 'precision', 'stability', 'business'
    
    Returns:
    --------
    dict con recomendación y justificación estadística
    """
    
    print(f"\n🎯 RECOMENDACIÓN ESTADÍSTICA PARA CASO: '{use_case.upper()}'")
    print("="*70)
    
    model_results = results['model_results']
    
    if use_case == 'precision':
        # Maximizar R² con menor margen de error
        criterion = lambda m: (model_results[m]['cv_r2']['mean'] * 
                              model_results[m]['confidence_score'])
        description = "Máxima precisión con alta confianza estadística"
        
    elif use_case == 'stability':
        # Maximizar estabilidad y minimizar ancho de IC
        criterion = lambda m: model_results[m]['cv_stability']
        description = "Máxima estabilidad y reproducibilidad"
        
    elif use_case == 'business':
        # Balance entre precisión y aplicabilidad práctica
        criterion = lambda m: (model_results[m]['cv_r2']['mean'] * 0.4 + 
                              model_results[m]['within_20pct']/100 * 0.4 +
                              model_results[m]['cv_stability'] * 0.2)
        description = "Mejor balance para aplicaciones de negocio"
        
    else:  # general
        # Score compuesto balanceado
        criterion = lambda m: (model_results[m]['cv_r2']['mean'] * 0.5 + 
                              model_results[m]['confidence_score'] * 0.3 +
                              model_results[m]['cv_stability'] * 0.2)
        description = "Mejor modelo general (precisión + confianza + estabilidad)"
    
    # Encontrar mejor modelo según criterio
    best_model = max(model_results.keys(), key=criterion)
    best_info = model_results[best_model]
    
    # Crear recomendación
    recommendation = {
        'recommended_model': best_model,
        'use_case': use_case,
        'description': description,
        'criterion_score': criterion(best_model),
        
        'statistical_evidence': {
            'cv_r2_mean': best_info['cv_r2']['mean'],
            'cv_r2_ci': [best_info['cv_r2']['ci_lower'], best_info['cv_r2']['ci_upper']],
            'cv_r2_margin_error': best_info['cv_r2']['margin_error'],
            'cv_rmse_mean': best_info['cv_rmse']['mean'],
            'cv_rmse_ci': [best_info['cv_rmse']['ci_lower'], best_info['cv_rmse']['ci_upper']],
            'stability_score': best_info['cv_stability'],
            'confidence_score': best_info['confidence_score'],
            'within_20pct': best_info['within_20pct']
        },
        
        'confidence_assessment': 'Alta' if best_info['confidence_score'] > 0.8 else 
                               'Media' if best_info['confidence_score'] > 0.6 else 'Baja'
    }
    
    # Mostrar recomendación
    print(f"🏆 Modelo recomendado: {best_model}")
    print(f"📝 Justificación: {description}")
    print(f"📊 Score del criterio: {criterion(best_model):.4f}")
    print(f"\n📈 Evidencia estadística:")
    print(f"   • R² CV: {best_info['cv_r2']['mean']:.4f} ± {best_info['cv_r2']['margin_error']:.4f}")
    print(f"   • RMSE CV: ${best_info['cv_rmse']['mean']:,.0f} ± ${best_info['cv_rmse']['margin_error']:,.0f}")
    print(f"   • Estabilidad: {best_info['cv_stability']:.3f}")
    print(f"   • Score de confianza: {best_info['confidence_score']:.3f}")
    print(f"   • Predicciones dentro ±20%: {best_info['within_20pct']:.1f}%")
    print(f"\n🎖️  Nivel de confianza: {recommendation['confidence_assessment']}")
    
    return recommendation


def analyze_feature_importance_with_uncertainty(X, feature_names, model, results, n_bootstrap=100):
    """
    Analizar importancia de features con intervalos de confianza usando bootstrap
    """
    
    if not hasattr(model, 'feature_importances_'):
        print("⚠️  El modelo no tiene feature_importances_")
        return None
    
    print(f"\n🎯 Analizando importancia de características con incertidumbre (n_bootstrap={n_bootstrap})...")
    
    # Importancia original
    original_importance = model.feature_importances_
    
    # Bootstrap para intervalos de confianza
    bootstrap_importances = []
    
    X_train = results['X_train']
    y_train = results['y_train']
    
    for i in range(n_bootstrap):
        # Muestra bootstrap
        n_samples = len(X_train)
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        X_bootstrap = X_train.iloc[bootstrap_indices] if hasattr(X_train, 'iloc') else X_train[bootstrap_indices]
        y_bootstrap = y_train.iloc[bootstrap_indices] if hasattr(y_train, 'iloc') else y_train[bootstrap_indices]
        
        # Entrenar modelo en muestra bootstrap
        bootstrap_model = type(model)(**model.get_params())
        bootstrap_model.fit(X_bootstrap, y_bootstrap)
        
        bootstrap_importances.append(bootstrap_model.feature_importances_)
    
    bootstrap_importances = np.array(bootstrap_importances)
    
    # Calcular estadísticas
    importance_stats = []
    for i, feature in enumerate(feature_names):
        feature_importances = bootstrap_importances[:, i]
        
        ci = calculate_confidence_intervals(feature_importances, 0.95)
        
        importance_stats.append({
            'feature': feature,
            'importance_mean': ci['mean'],
            'importance_original': original_importance[i],
            'ci_lower': ci['ci_lower'],
            'ci_upper': ci['ci_upper'],
            'margin_error': ci['margin_error'],
            'stability': 1 - (ci['std'] / max(abs(ci['mean']), 0.001))
        })
    
    # Convertir a DataFrame y ordenar
    importance_df = pd.DataFrame(importance_stats)
    importance_df = importance_df.sort_values('importance_mean', ascending=False)
    
    print("\nTOP 20 CARACTERÍSTICAS MÁS IMPORTANTES (con intervalos de confianza):")
    print("-" * 80)
    
    for i, row in importance_df.head(20).iterrows():
        print(f"{row['feature']:25} | "
              f"Imp: {row['importance_mean']:.4f} | "
              f"IC: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] | "
              f"Estab: {row['stability']:.3f}")
    
    # Visualización
    plt.figure(figsize=(12, 8))
    
    top_15 = importance_df.head(15)
    y_pos = range(len(top_15))
    
    # Barras con intervalos de confianza
    plt.barh(y_pos, top_15['importance_mean'], alpha=0.7, color='skyblue')
    
    # Intervalos de confianza
    for i, (lower, upper) in enumerate(zip(top_15['ci_lower'], top_15['ci_upper'])):
        plt.plot([lower, upper], [i, i], color='red', linewidth=2, marker='|', markersize=8)
    
    plt.yticks(y_pos, top_15['feature'])
    plt.xlabel('Importancia de Característica')
    plt.title('Top 15 Características con Intervalos de Confianza (95%)')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return importance_df


# Funciones auxiliares para compatibilidad con código existente
def analyze_predictions(results):
    """Función de compatibilidad - analizar calidad de predicciones"""
    if not results or 'model_results' not in results:
        print("❌ No hay resultados para analizar")
        return None
    
    print("\n📈 Analizando predicciones...")
    
    best_name = results['best_model_name']
    if best_name is None:
        print("❌ No hay mejor modelo seleccionado")
        return None
    
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
    
    # Intervalos de confianza si están disponibles
    if 'cv_r2' in best_result:
        r2_ci = best_result['cv_r2']
        print(f"\nIntervalos de confianza (95%):")
        print(f"   R² CV: {r2_ci['mean']:.3f} [{r2_ci['ci_lower']:.3f}, {r2_ci['ci_upper']:.3f}]")
    
    return analysis_df


def create_comparison_chart(results):
    """Función de compatibilidad - crear gráfico comparativo"""
    if not results or 'model_results' not in results:
        return None
    
    # Usar la nueva función de visualización
    create_confidence_interval_visualization(results)


# Función principal mejorada
def train_models(X, y, test_size=0.2, random_state=42, include_confidence_intervals=True, cv_folds=10):
    """
    Función principal compatible con el código existente pero con capacidades estadísticas mejoradas
    """
    
    if include_confidence_intervals:
        return train_models_with_confidence_intervals(
            X, y, test_size=test_size, random_state=random_state, cv_folds=cv_folds
        )
    else:
        # Versión original simplificada
        print("\n🚀 Entrenando modelos (versión básica)...")
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = get_enhanced_models(random_state)
        results = {}
        scaled_models = ['Linear Regression', 'Ridge Regression', 'SVR (RBF)', 'SVR (Linear)']
        
        for name, model in models.items():
            try:
                if name in scaled_models:
                    X_train_model = X_train_scaled
                    X_test_model = X_test_scaled
                else:
                    X_train_model = X_train
                    X_test_model = X_test
                
                model.fit(X_train_model, y_train)
                y_pred = model.predict(X_test_model)
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'rmse': rmse,
                    'r2': r2,
                    'mae': mae,
                    'predictions': y_pred,
                    'scaled': name in scaled_models
                }
                
            except Exception as e:
                print(f"Error entrenando {name}: {str(e)}")
                continue
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        best_model = results[best_model_name]['model']
        
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
        
        return final_results, best_model


def analyze_feature_importance(X, feature_names, model, include_uncertainty=True):
    """
    Función de compatibilidad mejorada para análisis de importancia
    """
    
    if include_uncertainty and hasattr(model, 'feature_importances_'):
        # Necesitamos acceso a los resultados para bootstrap
        # Por ahora usar versión básica
        pass
    
    # Versión básica original
    if not hasattr(model, 'feature_importances_'):
        print("⚠️  El modelo no tiene feature_importances_")
        return None
    
    print("\n🎯 Analizando importancia de características...")
    
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