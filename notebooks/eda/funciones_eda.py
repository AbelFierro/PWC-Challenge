import altair as alt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re

def load_data():
    """
    Carga y combina los archivos de datos CSV desde la carpeta data.
    
    Esta función lee tres archivos CSV (people.csv, salary.csv, descriptions.csv),
    los combina usando el campo 'id' como clave, y retorna un DataFrame unificado.
    
    Returns
    -------
    pandas.DataFrame or None
        DataFrame combinado con todos los datos si la carga es exitosa.
        None si ocurre algún error durante la carga.
    
    Notes
    -----
    Los archivos esperados son:
    - ../data/people.csv: Información demográfica
    - ../data/salary.csv: Información salarial  
    - ../data/descriptions.csv: Descripciones de empleados
    
    Examples
    --------
    >>> data = load_data()
    📁 Cargando datos...
    Datos cargados: 500 registros
          Columnas: ['id', 'Age', 'Gender', 'Education Level', 'Job Title', 'Salary', 'Description']
    """
    try:
        print("📁 Cargando datos...")
            
        # Cargar archivos
        people_df = pd.read_csv('../data/people.csv')
        salary_df = pd.read_csv('../data/salary.csv')
        descriptions_df = pd.read_csv('../data/descriptions.csv')
            
        # Combinar por ID
        data = people_df.merge(salary_df, on='id', how='inner')
        data = data.merge(descriptions_df, on='id', how='inner')
            
        print(f"Datos cargados: {len(data)} registros")
        print(f"      Columnas: {list(data.columns)}")
            
        return data
            
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return None

def imputar_edad(df):
    """
    Imputa valores faltantes de edad extrayendo información de la columna Description.
    
    Busca patrones de edad en texto como "25-year-old", "30 years old", o "I am 35"
    y asigna estos valores a filas donde Age es NaN.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame que contiene las columnas 'Age' y 'Description'.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame con valores de edad imputados donde fue posible.
    
    Notes
    -----
    Los patrones de búsqueda incluyen:
    - "X-year-old" (ej: "25-year-old")
    - "X years old" o "X year old" 
    - "I am X" (donde X es un número)
    
    Solo se modifican filas donde Age es NaN originalmente.
    
    Examples
    --------
    >>> df_imputado = imputar_edad(df)
    >>> # Verifica filas donde se imputó edad
    >>> mask_imputadas = df['Age'].isna() & (~df_imputado['Age'].isna())
    >>> print(f"Edades imputadas: {mask_imputadas.sum()}")
    """
    import re
    
    def extraer_edad(descripcion):
        if pd.isna(descripcion):
            return None
        match = re.search(r'(\d+)-year-old|(\d+)\s+years?\s+old|I\s+am\s+(\d+)', 
                         str(descripcion), re.IGNORECASE)
        if match:
            return int(next(g for g in match.groups() if g))
        return None
    
    # Aplicar solo a filas con age nulo
    mask = df['Age'].isna()
    df.loc[mask, 'Age'] = df.loc[mask, 'Description'].apply(extraer_edad)
    
    return df

def imputar_education_level(df, columna_education='Education Level', columna_descripcion='Description'):
    """
    Imputa valores faltantes de nivel educativo extrayendo información del texto descriptivo.
    
    Busca menciones de títulos educativos como "Bachelor's", "Master's", "PhD" y sus
    variaciones en la descripción, asignándolos a filas con valores faltantes.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame que contiene los datos.
    columna_education : str, default='Education Level'
        Nombre de la columna que contiene los niveles educativos.
    columna_descripcion : str, default='Description'
        Nombre de la columna con las descripciones de texto.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame con los niveles educativos imputados.
    
    Notes
    -----
    Patrones de búsqueda incluyen:
    - Bachelor's: "bachelor", "bachelor degree", "B.S.", "BS"
    - Master's: "master", "master degree", "M.S.", "MS", "MBA"
    - PhD: "PhD", "Ph.D.", "doctorate", "doctoral"
    
    La búsqueda es case-insensitive y busca coincidencias de palabras completas.
    
    Examples
    --------
    >>> df_imputado = imputar_education_level(df)
    Fila 45: Education level corregido a "Master's"
    Fila 78: Education level corregido a "Bachelor's"
    
    Total de correcciones realizadas: 12
    Valores nulos restantes en 'Education Level': 8
    """
    def extraer_education_level_de_descripcion(descripcion):
        """Extrae el nivel educativo de una descripción de texto"""
        if pd.isna(descripcion):
            return None
        
        # Niveles educativos a buscar
        niveles_educativos = ["Bachelor's", "Master's", 'PhD']
        
        # Convertir descripción a string para búsqueda
        texto = str(descripcion)
        
        # Buscar cada nivel educativo en el texto (case insensitive)
        for nivel in niveles_educativos:
            # Crear patrón flexible que capture variaciones
            patrones = [
                rf'\b{re.escape(nivel)}\b',  # Exacto
                rf'\b{re.escape(nivel.lower())}\b',  # Minúscula
                rf'\b{re.escape(nivel.upper())}\b',  # Mayúscula
            ]
            
            # Patrones adicionales para variaciones comunes
            if nivel == "Bachelor's":
                patrones.extend([
                    r'\bbachelor\s+degree\b',
                    r'\bb\.?s\.?\b',  # BS, B.S.
                    r'\bbachelor\b',
                ])
            elif nivel == "Master's":
                patrones.extend([
                    r'\bmaster\s+degree\b',
                    r'\bm\.?s\.?\b',  # MS, M.S.
                    r'\bmaster\b',
                    r'\bmba\b',  # MBA también es Master's
                ])
            elif nivel == 'PhD':
                patrones.extend([
                    r'\bph\.?d\.?\b',
                    r'\bdoctorate\b',
                    r'\bdoctoral\b',
                    r'\bphd\b',
                ])
            
            # Buscar cada patrón
            for patron in patrones:
                if re.search(patron, texto, re.IGNORECASE):
                    return nivel
        
        return None
    
    # Crear una copia del DataFrame para no modificar el original
    df_corregido = df.copy()
    
    # Encontrar filas donde education_level es nulo
    filas_nulas = df_corregido[columna_education].isna()
    
    # Contador para tracking
    correcciones = 0
    
    # Iterar sobre las filas con education_level nulo
    for idx in df_corregido[filas_nulas].index:
        descripcion = df_corregido.loc[idx, columna_descripcion]
        nivel_extraido = extraer_education_level_de_descripcion(descripcion)
        
        # Si se encontró un nivel educativo válido, asignarlo
        if nivel_extraido is not None:
            df_corregido.loc[idx, columna_education] = nivel_extraido
            correcciones += 1
            print(f"Fila {idx}: Education level corregido a '{nivel_extraido}'")
    
    print(f"\nTotal de correcciones realizadas: {correcciones}")
    print(f"Valores nulos restantes en '{columna_education}': {df_corregido[columna_education].isna().sum()}")
    
    return df_corregido

def imputar_gender(df, columna_gender='Gender', columna_descripcion='Description'):
    """
    Imputa valores faltantes de género extrayendo información del texto descriptivo.
    
    Busca pronombres y palabras clave que indiquen género (he/she, him/her, male/female)
    en la descripción y asigna 'Male' o 'Female' según corresponda.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame que contiene los datos.
    columna_gender : str, default='Gender'
        Nombre de la columna que contiene el género.
    columna_descripcion : str, default='Description'
        Nombre de la columna con las descripciones de texto.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame con los géneros imputados.
    
    Notes
    -----
    Patrones de búsqueda:
    - Female: 'female', 'woman', 'girl', 'she', 'her', 'hers', 'herself', 'lady'
    - Male: 'male', 'man', 'boy', 'he', 'him', 'his', 'himself', 'guy'
    
    La búsqueda es case-insensitive. Si se encuentran patrones de ambos géneros,
    se prioriza 'Female' (se puede ajustar el orden según necesidad).
    
    Examples
    --------
    >>> df_imputado = imputar_gender(df)
    Fila 23: Gender corregido a 'Female'
    Fila 67: Gender corregido a 'Male'
    
    Total de correcciones realizadas: 8
    Valores nulos restantes en 'Gender': 15
    """
    def extraer_gender_de_descripcion(descripcion):
        """Extrae el género de una descripción de texto"""
        if pd.isna(descripcion):
            return None
        
        # Convertir descripción a string para búsqueda
        texto = str(descripcion).lower()
        
        # Patrones para Female
        patrones_female = [
            r'\bfemale\b',
            r'\bwoman\b',
            r'\bgirl\b',
            r'\bshe\b',
            r'\bher\b',
            r'\bhers\b',
            r'\bherself\b',
            r'\blady\b',
        ]
        
        # Patrones para Male
        patrones_male = [
            r'\bmale\b',
            r'\bman\b',
            r'\bboy\b',
            r'\bhe\b',
            r'\bhim\b',
            r'\bhis\b',
            r'\bhimself\b',
            r'\bguy\b',
        ]
        
        # Buscar patrones Female
        for patron in patrones_female:
            if re.search(patron, texto):
                return 'Female'
        
        # Buscar patrones Male
        for patron in patrones_male:
            if re.search(patron, texto):
                return 'Male'
        
        return None
    
    # Crear una copia del DataFrame para no modificar el original
    df_corregido = df.copy()
    
    # Encontrar filas donde gender es nulo
    filas_nulas = df_corregido[columna_gender].isna()
    
    # Contador para tracking
    correcciones = 0
    
    # Iterar sobre las filas con gender nulo
    for idx in df_corregido[filas_nulas].index:
        descripcion = df_corregido.loc[idx, columna_descripcion]
        gender_extraido = extraer_gender_de_descripcion(descripcion)
        
        # Si se encontró un género válido, asignarlo
        if gender_extraido is not None:
            df_corregido.loc[idx, columna_gender] = gender_extraido
            correcciones += 1
            print(f"Fila {idx}: Gender corregido a '{gender_extraido}'")
    
    print(f"\nTotal de correcciones realizadas: {correcciones}")
    print(f"Valores nulos restantes en '{columna_gender}': {df_corregido[columna_gender].isna().sum()}")
    
    return df_corregido

def imputar_job_title(df, columna_job='Job Title', columna_descripcion='Description'):
    """
    Imputa valores faltantes de job title usando coincidencias parciales inteligentes.
    
    Busca coincidencias entre los job titles existentes y el texto de la descripción,
    priorizando títulos más específicos cuando hay múltiples coincidencias posibles.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame que contiene los datos.
    columna_job : str, default='Job Title'
        Nombre de la columna que contiene los títulos de trabajo.
    columna_descripcion : str, default='Description'
        Nombre de la columna con las descripciones de texto.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame con los job titles imputados.
    
    Notes
    -----
    Algoritmo de coincidencia:
    1. Divide cada job title en palabras clave
    2. Busca estas palabras en la descripción
    3. Calcula porcentaje de coincidencia por título
    4. Prioriza títulos más específicos (más palabras) cuando hay empates
    
    Ejemplo: "director of operations" coincidirá mejor con "Operations Director" 
    que con "Director" por mayor especificidad.
    
    Examples
    --------
    >>> df_imputado = imputar_job_title(df)
    Ejemplos de decisiones:
    ============================================================
    'I work as director of operations with extensive...' → 'Operations Director'
    'Business Intelligence Analyst with 5 years...' → 'Business Analyst'
    """
    job_titles_disponibles = df[columna_job].dropna().unique().tolist()
    
    def encontrar_mejor_coincidencia_priorizada(descripcion):
        if pd.isna(descripcion):
            return None
        
        texto = str(descripcion).lower()
        candidatos = []
        
        for job_title in job_titles_disponibles:
            palabras_job = job_title.lower().split()
            palabras_coincidentes = []
            
            # Encontrar palabras que coinciden
            for palabra in palabras_job:
                if len(palabra) > 2:  # Ignorar palabras cortas
                    if re.search(rf'\b{re.escape(palabra)}\b', texto):
                        palabras_coincidentes.append(palabra)
            
            # Calcular métricas de calidad
            if len(palabras_job) > 0 and len(palabras_coincidentes) > 0:
                porcentaje_coincidencia = len(palabras_coincidentes) / len(palabras_job)
                
                # Solo considerar si al menos 50% de las palabras coinciden
                if porcentaje_coincidencia >= 0.5:
                    candidatos.append({
                        'job_title': job_title,
                        'palabras_totales': len(palabras_job),
                        'palabras_coincidentes': len(palabras_coincidentes),
                        'porcentaje': porcentaje_coincidencia,
                        'especificidad': len(palabras_job)  # Más palabras = más específico
                    })
        
        if not candidatos:
            return None
        
        # Ordenar por criterios de prioridad:
        # 1. Porcentaje de coincidencia (descendente)
        # 2. Especificidad/número de palabras (descendente) 
        # 3. Palabras coincidentes absolutas (descendente)
        candidatos.sort(key=lambda x: (
            x['porcentaje'],           # Prioridad 1: % coincidencia
            x['especificidad'],        # Prioridad 2: más específico
            x['palabras_coincidentes'] # Prioridad 3: más coincidencias
        ), reverse=True)
        
        return candidatos[0]['job_title']
    
    # Aplicar correcciones
    df_corregido = df.copy()
    mask = df_corregido[columna_job].isna()
    
    print("Ejemplos de decisiones:")
    print("="*60)
    
    for idx in df_corregido[mask].index:
        descripcion = df_corregido.loc[idx, columna_descripcion]
        nuevo_job = encontrar_mejor_coincidencia_priorizada(descripcion)
        
        if nuevo_job is not None:
            df_corregido.loc[idx, columna_job] = nuevo_job
            print(f"'{descripcion[:50]}...' → '{nuevo_job}'")
    
    return df_corregido    

def crear_grafico_distribucion_avanzado(data, columna_salary='Salary', 
                                       maxbins=20, titulo=None, 
                                       ancho=400, alto=300, 
                                       color='steelblue'):
    """
    Crea un histograma avanzado de distribución de una variable numérica.
    
    Genera un gráfico de barras con opciones de personalización que muestra
    la distribución de frecuencias de una variable, incluyendo estadísticas
    en el título y tooltips interactivos.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame que contiene los datos.
    columna_salary : str, default='Salary'
        Nombre de la columna numérica a analizar.
    maxbins : int, default=20
        Número máximo de bins para agrupar los valores.
    titulo : str, optional
        Título personalizado del gráfico. Si None, se genera automáticamente.
    ancho : int, default=400
        Ancho del gráfico en píxeles.
    alto : int, default=300
        Alto del gráfico en píxeles.
    color : str, default='steelblue'
        Color de las barras.
    
    Returns
    -------
    altair.Chart
        Gráfico de Altair con el histograma de distribución.
    
    Raises
    ------
    ValueError
        Si la columna especificada no existe o no hay datos válidos.
    
    Examples
    --------
    >>> chart = crear_grafico_distribucion_avanzado(data, 'Salary')
    >>> chart.show()
    
    >>> # Con personalización
    >>> chart = crear_grafico_distribucion_avanzado(
    ...     data, 'Age', maxbins=15, titulo='Distribución de Edades',
    ...     color='darkgreen', ancho=600
    ... )
    """
    # Validaciones
    if columna_salary not in data.columns:
        raise ValueError(f"La columna '{columna_salary}' no existe en el DataFrame")
    
    df_clean = data.dropna(subset=[columna_salary])
    
    if df_clean.empty:
        raise ValueError(f"No hay datos válidos en la columna '{columna_salary}'")
    
    if titulo is None:
        titulo = f'Distribución de {columna_salary}'
    
    # Información adicional para el título
    n_registros = len(df_clean)
    salario_promedio = df_clean[columna_salary].mean()
    
    titulo_completo = f'{titulo}\n(n={n_registros:,}, promedio=${salario_promedio:,.0f})'
    
    # Crear gráfico
    chart = alt.Chart(df_clean).mark_bar(
        color=color,
        opacity=0.7,
        stroke='white',
        strokeWidth=1
    ).encode(
        alt.X(f'{columna_salary}:Q', 
              bin=alt.Bin(maxbins=maxbins),
              title=f'{columna_salary} (USD)',
              axis=alt.Axis(format='$,.0f')),
        alt.Y('count()', 
              title='Número de empleados'),
        tooltip=[
            alt.Tooltip(f'{columna_salary}:Q', bin=True, title='Rango salarial'),
            alt.Tooltip('count()', title='Cantidad')
        ]
    ).properties(
        title=titulo_completo,
        width=ancho,
        height=alto
    )
    
    return chart

def crear_heatmap_correlacion(data, titulo='Matriz de Correlación', 
                             ancho=400, alto=400, esquema_color='redblue'):
    """
    Crea un heatmap de correlación entre variables numéricas.
    
    Genera una matriz de correlación visual que muestra las relaciones
    entre todas las variables numéricas del dataset, con valores de
    correlación superpuestos y tooltips informativos.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame que contiene los datos numéricos.
    titulo : str, default='Matriz de Correlación'
        Título del heatmap.
    ancho : int, default=400
        Ancho del gráfico en píxeles.
    alto : int, default=400
        Alto del gráfico en píxeles.
    esquema_color : str, default='category10'
        Esquema de colores de Altair ('category10', 'viridis', 'set1', etc.).
    
    Returns
    -------
    altair.Chart
        Gráfico boxplot avanzado con capas opcionales.
    
    Raises
    ------
    ValueError
        Si las columnas especificadas no existen o no hay datos válidos.
    
    Examples
    --------
    >>> chart = crear_boxplot_avanzado(
    ...     data, 'Education Level', 'Salary',
    ...     mostrar_puntos=True, esquema_color='viridis'
    ... )
    >>> chart.show()
    """
    # Validaciones
    if columna_categorica not in data.columns:
        raise ValueError(f"La columna '{columna_categorica}' no existe en el DataFrame")
    
    if columna_numerica not in data.columns:
        raise ValueError(f"La columna '{columna_numerica}' no existe en el DataFrame")
    
    # Limpiar datos
    df_clean = data.dropna(subset=[columna_categorica, columna_numerica])
    
    if df_clean.empty:
        raise ValueError(f"No hay datos válidos")
    
    # Título por defecto
    if titulo is None:
        titulo = f'Análisis de {columna_numerica} por {columna_categorica}'
    
    # Base chart
    base = alt.Chart(df_clean)
    
    # Boxplot principal
    boxplot = base.mark_boxplot(
        extent='min-max' if mostrar_outliers else 'ci',
        size=60
    ).encode(
        alt.X(f'{columna_categorica}:N', 
              title=columna_categorica,
              axis=alt.Axis(labelAngle=-45, labelFontSize=11)),
        alt.Y(f'{columna_numerica}:Q', 
              title=columna_numerica,
              axis=alt.Axis(format='.0f')),
        alt.Color(f'{columna_categorica}:N',
                  scale=alt.Scale(scheme=esquema_color),
                  legend=alt.Legend(
                      title=columna_categorica,
                      orient='right'
                  )),
        tooltip=[
            alt.Tooltip(f'{columna_categorica}:N', title='Categoría'),
            alt.Tooltip(f'{columna_numerica}:Q', title='Valor', format='.0f')
        ]
    )
    
    # Capas adicionales
    layers = [boxplot]
    
    # Agregar puntos si se solicita
    if mostrar_puntos:
        points = base.mark_circle(
            opacity=0.3,
            size=30
        ).encode(
            alt.X(f'{columna_categorica}:N'),
            alt.Y(f'{columna_numerica}:Q'),
            alt.Color(f'{columna_categorica}:N',
                      scale=alt.Scale(scheme=esquema_color),
                      legend=None)
        )
        layers.append(points)
    
    # Combinar capas
    chart = alt.layer(*layers).resolve_scale(
        color='independent'
    ).properties(
        title=alt.TitleParams(
            text=titulo,
            fontSize=14,
            anchor='start'
        ),
        width=ancho,
        height=alto
    )
    
    return chart

def crear_boxplot_con_estadisticas(data, columna_categorica, columna_numerica,
                                  titulo=None, mostrar_n=True):
    """
    Crea un boxplot que incluye estadísticas descriptivas en las etiquetas.
    
    Genera un boxplot con información estadística adicional, incluyendo
    el número de observaciones por categoría y estadísticas resumidas
    que facilitan la interpretación de los datos.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame que contiene los datos.
    columna_categorica : str
        Nombre de la columna categórica (eje X).
    columna_numerica : str
        Nombre de la columna numérica (eje Y).
    titulo : str, optional
        Título personalizado del gráfico.
    mostrar_n : bool, default=True
        Si mostrar el número de observaciones por categoría en las etiquetas.
    
    Returns
    -------
    tuple
        Tupla con (altair.Chart, pandas.DataFrame) donde el DataFrame
        contiene las estadísticas descriptivas por categoría.
    
    Raises
    ------
    ValueError
        Si las columnas especificadas no existen o no hay datos válidos.
    
    Examples
    --------
    >>> chart, stats = crear_boxplot_con_estadisticas(
    ...     data, 'Education Level', 'Salary'
    ... )
    >>> chart.show()
    >>> print(stats)  # Muestra estadísticas por categoría
                count    mean  median     std
    Bachelor's    150  65000   62000  12000
    Master's      100  85000   82000  15000
    PhD            50 105000  100000  18000
    """
    # Validaciones y limpieza
    if columna_categorica not in data.columns:
        raise ValueError(f"La columna '{columna_categorica}' no existe")
    
    if columna_numerica not in data.columns:
        raise ValueError(f"La columna '{columna_numerica}' no existe")
    
    df_clean = data.dropna(subset=[columna_categorica, columna_numerica])
    
    if df_clean.empty:
        raise ValueError("No hay datos válidos")
    
    # Calcular estadísticas por categoría
    stats = df_clean.groupby(columna_categorica)[columna_numerica].agg([
        'count', 'mean', 'median', 'std'
    ]).round(2)
    
    # Título con estadísticas
    if titulo is None:
        titulo = f'Distribución de {columna_numerica} por {columna_categorica}'
        if mostrar_n:
            total_obs = len(df_clean)
            titulo += f'\n(Total observaciones: {total_obs:,})'
    
    # Crear etiquetas con N por categoría si se solicita
    if mostrar_n:
        # Agregar columna con N al DataFrame
        df_with_n = df_clean.copy()
        category_counts = df_clean[columna_categorica].value_counts()
        df_with_n['category_label'] = df_with_n[columna_categorica].map(
            lambda x: f"{x}\n(n={category_counts[x]})"
        )
        x_column = 'category_label:N'
    else:
        df_with_n = df_clean
        x_column = f'{columna_categorica}:N'
    
    # Crear boxplot
    chart = alt.Chart(df_with_n).mark_boxplot(extent='min-max').encode(
        alt.X(x_column, 
              title=columna_categorica,
              axis=alt.Axis(labelAngle=-45 if mostrar_n else 0)),
        alt.Y(f'{columna_numerica}:Q', 
              title=columna_numerica,
              axis=alt.Axis(format='.0f')),
        alt.Color(f'{columna_categorica}:N',
                  legend=alt.Legend(title=columna_categorica)),
        tooltip=[
            alt.Tooltip(f'{columna_categorica}:N', title='Categoría'),
            alt.Tooltip(f'{columna_numerica}:Q', title='Valor', format='.0f')
        ]
    ).properties(
        title=titulo,
        width=500,
        height=400
    )
    
    return chart, stats



def aplicar_kmeans_clustering(data, columnas_features, n_clusters=3, 
                             random_state=42, normalizar=True, 
                             nombre_cluster='Cluster'):
    """
    Aplica K-means clustering a las columnas especificadas
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame que contiene los datos
    columnas_features : list
        Lista de nombres de columnas numéricas para el clustering
    n_clusters : int, default=3
        Número de clusters a crear
    random_state : int, default=42
        Semilla para reproducibilidad
    normalizar : bool, default=True
        Si normalizar las variables antes del clustering
    nombre_cluster : str, default='Cluster'
        Nombre de la columna donde guardar los clusters
    
    Returns:
    --------
    tuple: (DataFrame con clusters, modelo KMeans, scaler si se usó)
        DataFrame modificado con la columna de clusters
        Modelo KMeans entrenado
        Scaler usado (None si normalizar=False)
    
    Example:
    --------
    >>> df_con_clusters, modelo, scaler = aplicar_kmeans_clustering(
    ...     data, ['Age', 'Years of Experience', 'Salary'], n_clusters=3
    ... )
    """
    
    # Validar que las columnas existen
    for col in columnas_features:
        if col not in data.columns:
            raise ValueError(f"La columna '{col}' no existe en el DataFrame")
    
    # Extraer datos y eliminar NaNs
    X = data[columnas_features].dropna()
    
    if X.empty:
        raise ValueError("No hay datos válidos después de eliminar NaNs")
    
    print(f"Aplicando K-means con {n_clusters} clusters")
    print(f"Datos utilizados: {len(X)} filas, {len(columnas_features)} características")
    print(f"Características: {columnas_features}")
    
    # Normalizar datos si se solicita
    scaler = None
    if normalizar:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("✓ Datos normalizados")
    else:
        X_scaled = X.values
        print("✓ Datos sin normalizar")
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Crear copia del DataFrame original
    data_con_clusters = data.copy()
    
    # Asignar clusters solo a las filas que tenían datos válidos
    data_con_clusters.loc[X.index, nombre_cluster] = clusters
    
    # Información sobre la asignación
    filas_con_cluster = (~data_con_clusters[nombre_cluster].isna()).sum()
    filas_sin_cluster = data_con_clusters[nombre_cluster].isna().sum()
    
    print(f"✓ Columna '{nombre_cluster}' agregada al DataFrame")
    print(f"   Filas con cluster asignado: {filas_con_cluster}")
    print(f"   Filas sin cluster (por NaN): {filas_sin_cluster}")
    
    # Estadísticas del clustering
    print(f"\n📊 Resultados del clustering:")
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        porcentaje = (count / len(clusters)) * 100
        print(f"   Cluster {cluster_id}: {count} observaciones ({porcentaje:.1f}%)")
    
    print(f"   Inercia (WCSS): {kmeans.inertia_:.2f}")
    
    return data_con_clusters, kmeans, scaler

def visualizar_clusters_2d(data_con_clusters, col_x, col_y, 
                          nombre_cluster='Cluster', titulo=None,
                          ancho=500, alto=400):
    """
    Visualiza clusters en un gráfico 2D
    
    Parameters:
    -----------
    data_con_clusters : pandas.DataFrame
        DataFrame que contiene los clusters
    col_x : str
        Columna para el eje X
    col_y : str
        Columna para el eje Y  
    nombre_cluster : str, default='Cluster'
        Nombre de la columna de clusters
    titulo : str, optional
        Título del gráfico
    """
    
    # Filtrar solo filas con clusters asignados
    df_plot = data_con_clusters.dropna(subset=[col_x, col_y, nombre_cluster])
    
    if df_plot.empty:
        raise ValueError("No hay datos válidos para visualizar")
    
    if titulo is None:
        titulo = f'Clusters K-means: {col_y} vs {col_x}'
    
    # Convertir cluster a string para mejor visualización
    df_plot = df_plot.copy()
    df_plot[f'{nombre_cluster}_str'] = df_plot[nombre_cluster].astype(str)
    
    chart = alt.Chart(df_plot).mark_circle(
        size=80,
        opacity=0.7,
        stroke='white',
        strokeWidth=1
    ).encode(
        alt.X(f'{col_x}:Q', 
              title=col_x,
              axis=alt.Axis(format='.0f')),
        alt.Y(f'{col_y}:Q', 
              title=col_y,
              axis=alt.Axis(format='.0f')),
        alt.Color(f'{nombre_cluster}_str:N',
                  scale=alt.Scale(scheme='category10'),
                  legend=alt.Legend(title='Cluster')),
        tooltip=[
            alt.Tooltip(f'{col_x}:Q', title=col_x, format='.0f'),
            alt.Tooltip(f'{col_y}:Q', title=col_y, format='.0f'),
            alt.Tooltip(f'{nombre_cluster}_str:N', title='Cluster')
        ]
    ).properties(
        title=titulo,
        width=ancho,
        height=alto
    )
    
    return chart


def agregar_clusters_a_dataframe(data, columnas_features, n_clusters=3,
                                 random_state=42, normalizar=True,
                                 nombre_cluster='Cluster', inplace=False):
    """
    Función específica para agregar clusters al DataFrame existente
    
    Parameters:
    -----------
    inplace : bool, default=False
        Si True, modifica el DataFrame original. Si False, retorna una copia.
    
    Returns:
    --------
    pandas.DataFrame o None
        DataFrame con clusters agregados (si inplace=False) o None (si inplace=True)
    
    Example:
    --------
    >>> # Modificar DataFrame original
    >>> agregar_clusters_a_dataframe(data, ['Age', 'Salary'], inplace=True)
    >>> 
    >>> # O crear nueva versión
    >>> data_nueva = agregar_clusters_a_dataframe(data, ['Age', 'Salary'])
    """
    
    print(f"🔄 Agregando clusters al DataFrame...")
    
    # Aplicar clustering
    data_con_clusters, modelo, scaler = aplicar_kmeans_clustering(
        data, columnas_features, n_clusters, random_state, normalizar, nombre_cluster
    )
    
    if inplace:
        # Modificar DataFrame original
        data[nombre_cluster] = data_con_clusters[nombre_cluster]
        print(f"✅ Columna '{nombre_cluster}' agregada al DataFrame original")
        return None
    else:
        # Retornar nueva versión
        print(f"✅ DataFrame con clusters retornado (original sin modificar)")
        return data_con_clusters