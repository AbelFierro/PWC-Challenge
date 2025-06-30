import funciones_lgbm as f_lgbm
import pandas as pd
import numpy as np


#===============================================


def calculate_groups(age, years_of_experience, grouping_info=None):
    """
    Calcula Age_group y Exp_group automáticamente
    """
    # Configuración por defecto si no hay grouping_info
    default_age_bins = [22, 32, 38, 46, float('inf')]
    default_age_labels = ['Joven', 'Adulto_Joven', 'Adulto_Medio', 'Senior']
    
    default_exp_bins = [0, 5, 15, float('inf')]
    default_exp_labels = ['Junior', 'Medio', 'Senior']
    
    # Usar configuración del modelo si está disponible
    if grouping_info:
        age_bins = grouping_info.get('age_bins', default_age_bins)
        age_labels = grouping_info.get('age_labels', default_age_labels)
        exp_bins = grouping_info.get('exp_bins', default_exp_bins)
        exp_labels = grouping_info.get('exp_labels', default_exp_labels)
    else:
        age_bins = default_age_bins
        age_labels = default_age_labels
        exp_bins = default_exp_bins
        exp_labels = default_exp_labels
    
    # Calcular Age_group
    age_group = age_labels[-1]  # Por defecto el último grupo
    for i in range(len(age_bins) - 1):
        if age_bins[i] <= age < age_bins[i + 1]:
            age_group = age_labels[i]
            break
    
    # Calcular Exp_group
    exp_group = exp_labels[-1]  # Por defecto el último grupo
    for i in range(len(exp_bins) - 1):
        if exp_bins[i] <= years_of_experience < exp_bins[i + 1]:
            exp_group = exp_labels[i]
            break
    
    return exp_group, age_group


#===============================================

def create_statistical_features_pred(data, stats_dict):
    """
    Versión adaptada para un solo registro que maneja features que requieren múltiples registros
    """
    print(f"📊 Creando features estadísticos para un solo registro...")
    
    # Verificar que es un solo registro
    if len(data) != 1:
        print(f"⚠️  Esta función es para un solo registro, recibido: {len(data)}")
    
    features = pd.DataFrame(index=data.index)
    
    # ============= FEATURES QUE SÍ FUNCIONAN CON 1 REGISTRO =============
    
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
    
    # ============= FEATURES QUE NO FUNCIONAN CON 1 REGISTRO =============
    # Estas requieren múltiples registros para calcular rankings/percentiles
    
    # SOLUCIÓN: Usar valores por defecto o aproximaciones
    
    # En lugar de rankings reales, usar percentiles aproximados basados en z-scores
    age_features = [col for col in features.columns if 'age_zscore_' in col]
    exp_features = [col for col in features.columns if 'exp_zscore_' in col]
    
    if age_features:
        # Simular rankings basados en z-scores (percentiles aproximados)
        features['age_rank_in_edu_group'] = scipy_stats.norm.cdf(features.get('age_zscore_vs_edu', 0))
        features['age_rank_in_gender'] = scipy_stats.norm.cdf(features.get('age_zscore_vs_gender', 0))
        features['age_rank_in_edu_gender'] = scipy_stats.norm.cdf(features.get('age_zscore_vs_edu', 0))  # Aproximación
    else:
        features['age_rank_in_edu_group'] = 0.5  # Valor neutral
        features['age_rank_in_gender'] = 0.5
        features['age_rank_in_edu_gender'] = 0.5
    
    if exp_features:
        features['exp_rank_in_edu_group'] = scipy_stats.norm.cdf(features.get('exp_zscore_vs_edu', 0))
        features['exp_rank_in_gender'] = scipy_stats.norm.cdf(features.get('exp_zscore_vs_gender', 0))
        features['exp_rank_in_edu_gender'] = scipy_stats.norm.cdf(features.get('exp_zscore_vs_edu', 0))  # Aproximación
    else:
        features['exp_rank_in_edu_group'] = 0.5
        features['exp_rank_in_gender'] = 0.5
        features['exp_rank_in_edu_gender'] = 0.5
    
    
    age_comparison_features = [col for col in features.columns if 'age_vs_' in col]
    exp_comparison_features = [col for col in features.columns if 'exp_vs_' in col]
    
    if age_comparison_features:
        
        features['age_consistency_score'] = features[age_comparison_features].std(axis=1)
    else:
        features['age_consistency_score'] = 0
    
    if exp_comparison_features:
        features['exp_consistency_score'] = features[exp_comparison_features].std(axis=1)
    else:
        features['exp_consistency_score'] = 0
    
    # Limpiar NaN
    features = features.fillna(0)
    
    print(f"   ✅ Creadas {len(features.columns)} features estadísticos para un solo registro")
    
    return features

def create_features_with_stats_pred(data, all_job_categories, all_seniority_levels, stats_dict):
    """
    Versión para un solo registro que garantiza el mismo número de features que en entrenamiento
    """
    print("🔧 Creando características completas para un solo registro...")
    
    # Verificar que no hay Salary
    if 'Salary' in data.columns:
        raise ValueError("❌ ERROR: 'Salary' encontrado en data. Usar data sin columna target.")
    
    # 1. Crear features originales
    original_features, feature_names = f_lgbm.create_features(data, all_job_categories, all_seniority_levels)
    
    # 2. Crear features estadísticos adaptados para un solo registro
    stat_features = create_statistical_features_pred(data, stats_dict)
    
    # 3. Combinar
    combined_features = pd.concat([original_features, stat_features], axis=1)
    combined_feature_names = list(combined_features.columns)
    
    print(f"✅ Features totales para un solo registro: {len(combined_feature_names)}")
    print(f"   - Originales: {len(feature_names)}")
    print(f"   - Estadísticos: {len(stat_features.columns)}")
    
    return combined_features, combined_feature_names



def predict(new_data, model_package):
    """
    Función de predicción:
    - Modelo optimizado 
    - Ensembles 
    """
    print("🎯 Predicción con detección automática de tipo de modelo...")
    
    
    print("🔍 === DEBUG COMPLETO ===")
    print(f"🔍 Claves en model_package: {list(model_package.keys())}")
    print(f"🔍 is_ensemble: {model_package.get('is_ensemble')}")
    print(f"🔍 Tipo de model_package['model']: {type(model_package['model'])}")
    
    # Si es dict, mostrar contenido
    if isinstance(model_package.get('model'), dict):
        print(f"🔍 model es dict con claves: {list(model_package['model'].keys())}")
    
    # Verificar la condición del if
    is_ensemble_condition = model_package.get('is_ensemble', False)
    print(f"🔍 Condición is_ensemble evalúa a: {is_ensemble_condition}")
    
    if is_ensemble_condition:
        print("🎯 ENTRANDO AL BLOQUE DE ENSEMBLE")
    else:
        print("🤖 ENTRANDO AL BLOQUE DE MODELO NORMAL")
    
    # DETECTAR SI ES ENSEMBLE
    if model_package.get('is_ensemble', False):
        print("🎯 Detectado ENSEMBLE, usando predicción especial...")
        
        # Verificar que tiene los componentes necesarios
        ensemble_components = model_package.get('model', {})
        if not isinstance(ensemble_components, dict):
            print("❌ Ensemble mal formateado")
            return None
        
        individual_models = ensemble_components.get('individual_models', {})
        weights = ensemble_components.get('weights', None)
        scaler = ensemble_components.get('scaler', None)
        
        if not individual_models:
            print("❌ No hay modelos individuales en el ensemble")
            return None
        
        # Crear grupos si no existen
        input_data_copy = new_data.copy()
        if 'Exp_group' not in input_data_copy.columns or 'Age_group' not in input_data_copy.columns:
            for idx, row in input_data_copy.iterrows():
                exp_group, age_group = calculate_groups(
                    age=row['Age'], 
                    years_of_experience=row['Years_of_Experience'], 
                    grouping_info=model_package['grouping_info']
                )
                input_data_copy.at[idx, 'Exp_group'] = exp_group
                input_data_copy.at[idx, 'Age_group'] = age_group
        
        # Crear features
        X_features, _ = create_features_with_stats_pred(
            input_data_copy,
            all_job_categories=model_package['job_categories'],
            all_seniority_levels=model_package['seniority_categories'],
            stats_dict=model_package['stats_dict']
        )
        
        # Hacer predicciones con cada modelo individual
        predictions = []
        valid_weights = []
        
        for i, (name, model) in enumerate(individual_models.items()):
            try:
                if name in ['Ridge', 'ElasticNet'] and scaler is not None:
                    X_scaled = scaler.transform(X_features)
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X_features)
                
                predictions.append(pred[0] if len(pred) == 1 else pred)
                
                # Agregar peso correspondiente
                if weights is not None and i < len(weights):
                    valid_weights.append(weights[i])
                
            except Exception as e:
                print(f"⚠️ Error prediciendo con {name}: {e}")
                continue
        
        if not predictions:
            print("❌ No se pudieron hacer predicciones con ningún modelo")
            return None
        
        # Combinar predicciones
        if valid_weights and len(valid_weights) == len(predictions):
            # Usar pesos si están disponibles y coinciden
            final_prediction = np.average(predictions, weights=valid_weights)
        else:
            # Promedio simple
            final_prediction = np.mean(predictions)
        
        print(f"   💰 Predicción ensemble: ${final_prediction:,.2f}")
        print(f"   🤖 Modelos usados: {len(predictions)}")
        
        return final_prediction
    
    else:
        print("🤖 Detectado modelo NORMAL, usando predicción estándar...")
        
        # VERIFICAR QUE model_package['model'] ES REALMENTE UN MODELO
        actual_model = model_package.get('model')
        if actual_model is None:
            print("❌ No hay modelo en model_package")
            return None
        
        # Verificar que es un modelo y no un diccionario
        if isinstance(actual_model, dict):
            print("❌ Error: model_package['model'] es un diccionario, no un modelo")
            print(f"   Claves encontradas: {list(actual_model.keys())}")
            return None
        
        # Verificar que tiene método predict
        if not hasattr(actual_model, 'predict'):
            print(f"❌ Error: El objeto no tiene método 'predict'. Tipo: {type(actual_model)}")
            return None
        
        
        # Verificar que el modelo tiene features estadísticos
        if not model_package.get('has_statistical_features', False):
            print("⚠️  Este modelo no tiene features estadísticos")
            print("❌ Función predict_salary_standard no implementada")
            return None
        
        # Verificar dimensiones esperadas
        expected_features = model_package.get('total_features')
        if expected_features is None:
            print("❌ Error: 'total_features' no encontrado en model_package")
            return None
            
        print(f"   🔢 Features esperadas: {expected_features}")
        
        # Crear features usando la versión para un solo registro
        try:
            X_new, feature_names = create_features_with_stats_pred(
                new_data,
                all_job_categories=model_package['job_categories'],
                all_seniority_levels=model_package['seniority_categories'],
                stats_dict=model_package['stats_dict']
            )
        except Exception as e:
            print(f"❌ Error creando features: {e}")
            return None
        
        print(f"   🔢 Features generadas: {len(feature_names)}")
        
        # Verificar dimensiones
        if len(feature_names) != expected_features:
            print(f"   ⚠️  Ajustando dimensiones: {len(feature_names)} → {expected_features}")
            
            # Alinear con features del modelo
            model_feature_names = model_package.get('feature_names', [])
            if not model_feature_names:
                print("❌ Error: 'feature_names' no encontrado en model_package")
                return None
                
            X_aligned = pd.DataFrame(0, index=X_new.index, columns=model_feature_names)
            
            # Llenar con los valores disponibles
            for col in X_new.columns:
                if col in X_aligned.columns:
                    X_aligned[col] = X_new[col]
            
            X_new = X_aligned
            print(f"   ✅ Dimensiones alineadas: {X_new.shape}")
        
        # Predecir
        try:
            prediction = actual_model.predict(X_new)[0]
            
            print(f"   💰 Predicción: ${prediction:,.2f}")
            print(f"   ✅ Predicción exitosa con {X_new.shape[1]} features")
            
            return prediction
            
        except Exception as e:
            print(f"❌ Error en predicción del modelo: {e}")
            return None