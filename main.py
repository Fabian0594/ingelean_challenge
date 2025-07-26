"""
Análisis de Datos de Talento - Challenge Ingelean
================================================

Este módulo contiene funciones para el preprocesamiento y análisis de correlaciones 
de un dataset de manufactura. Genera un dataset limpio listo para machine learning.

Para modelos de árbol de decisión, ejecutar: python decision_tree.py

Autores: 
Fecha: 2025
"""

import os
from dotenv import load_dotenv
from process_data import DataProcessor
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# from ydata_profiling import ProfileReport  # Usamos ydata-profiling que es compatible con Python 3.11

# ================================================================================================
# FUNCIONES DE PREPROCESAMIENTO DE DATOS
# ================================================================================================

def crear_fallos_binarios(df, columna_fallo='tipo_fallo'):
    """
    Crea una columna binaria de fallos basada en valores NaN.
    
    Esta función transforma la información de fallos en una variable binaria donde:
    - 0 indica ausencia de fallo (valor NaN en la columna original)
    - 1 indica presencia de fallo (valor no-NaN en la columna original)
    
    Args:
        df (pd.DataFrame): DataFrame original con los datos
        columna_fallo (str, optional): Nombre de la columna a evaluar para crear 
                                     la variable binaria. Por defecto 'tipo_fallo'
    
    Returns:
        pd.DataFrame: DataFrame con una nueva columna 'fallos_binarios' agregada
        
    Example:
        >>> df_con_fallos = crear_fallos_binarios(df, 'tipo_fallo')
        >>> print(df_con_fallos['fallos_binarios'].value_counts())
    """
    df_con_binarios = df.copy()
    
    # Crear columna binaria: 1 si NO es NaN (hay fallo), 0 si es NaN (sin fallo)
    df_con_binarios['fallos_binarios'] = (~df_con_binarios[columna_fallo].isna()).astype(int)
    
    print(f"\nColumna 'fallos_binarios' creada:")
    print(f"- Sin fallo (0): {(df_con_binarios['fallos_binarios'] == 0).sum()} registros")
    print(f"- Con fallo (1): {(df_con_binarios['fallos_binarios'] == 1).sum()} registros")
    
    return df_con_binarios


def imputacion_simple(df, estrategia='media'):
    """
    Aplica imputación simple a las columnas numéricas del DataFrame.
    
    Esta función rellena los valores faltantes en las columnas numéricas usando
    la estrategia especificada (media o mediana).
    
    Args:
        df (pd.DataFrame): DataFrame con valores faltantes
        estrategia (str, optional): Estrategia de imputación. Opciones:
                                  - 'media': Usar la media de la columna
                                  - 'mediana': Usar la mediana de la columna
                                  Por defecto 'media'
    
    Returns:
        pd.DataFrame: DataFrame con valores faltantes imputados en columnas numéricas
        
    Note:
        Solo se aplica imputación a columnas de tipo numérico (int64, float64, int32, float32).
        Las columnas categóricas y de texto no son modificadas.
        
    Example:
        >>> df_imputado = imputacion_simple(df, estrategia='media')
        >>> print(df_imputado.isnull().sum())
    """
    # Crear una copia del DataFrame original
    df_imputado = df.copy()
    
    # Identificar columnas numéricas
    columnas_numericas = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    
    # Aplicar imputación solo a columnas numéricas
    if estrategia == 'media':
        df_imputado[columnas_numericas] = df_imputado[columnas_numericas].fillna(
            df_imputado[columnas_numericas].mean()
        )
    else:
        df_imputado[columnas_numericas] = df_imputado[columnas_numericas].fillna(
            df_imputado[columnas_numericas].median()
        )
    
    return df_imputado


# ================================================================================================
# FUNCIONES DE ANÁLISIS DE CORRELACIONES
# ================================================================================================

def pearson_correlation(df, col1, col2):
    """
    Calcula la correlación de Pearson entre dos columnas específicas.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene las columnas
        col1 (str): Nombre de la primera columna
        col2 (str): Nombre de la segunda columna
        
    Returns:
        float: Coeficiente de correlación de Pearson entre las dos columnas
        
    Raises:
        KeyError: Si alguna de las columnas no existe en el DataFrame
        
    Example:
        >>> corr = pearson_correlation(df, 'temperatura', 'vibracion')
        >>> print(f"Correlación: {corr:.4f}")
    """
    return df[[col1, col2]].corr().iloc[0, 1]


def crear_tabla_correlaciones(df, guardar_archivo=True, mostrar_heatmap=True):
    """
    Genera análisis completo de correlaciones de Pearson para variables numéricas.
    
    Esta función calcula la matriz de correlaciones, identifica las correlaciones
    más significativas, y genera archivos de salida para análisis posterior.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos a analizar
        guardar_archivo (bool, optional): Si True, guarda los resultados en archivos CSV.
                                        Por defecto True
        mostrar_heatmap (bool, optional): Si True, genera y guarda un mapa de calor.
                                        Por defecto True
    
    Returns:
        pd.DataFrame: Matriz de correlaciones de Pearson (n x n) donde n es el 
                     número de variables numéricas
        
    Files Generated:
        - matriz_correlaciones.csv: Matriz completa de correlaciones
        - top_correlaciones.csv: Correlaciones ordenadas por valor absoluto (2 decimales)
        - heatmap_correlaciones.png: Visualización en mapa de calor
        
    Note:
        - Solo considera variables numéricas (int64, float64, int32, float32)
        - Las correlaciones en el archivo CSV se redondean a 2 decimales
        - El mapa de calor usa escala de colores coolwarm centrada en 0
        
    Example:
        >>> matriz = crear_tabla_correlaciones(df_limpio)
        >>> print(f"Variables analizadas: {matriz.shape[0]}")
    """
    # Seleccionar solo columnas numéricas
    columnas_numericas = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    
    if len(columnas_numericas) == 0:
        print("❌ No se encontraron columnas numéricas para calcular correlaciones.")
        return None
    
    print(f"\n🔍 Calculando correlaciones de Pearson para {len(columnas_numericas)} variables numéricas...")
    
    # Calcular matriz de correlación
    matriz_correlacion = df[columnas_numericas].corr(method='pearson')
    
    # Mostrar información básica
    print("✅ Matriz de correlaciones calculada exitosamente")
    
    # Identificar correlaciones más altas (excluyendo la diagonal)
    mask = np.triu(np.ones_like(matriz_correlacion, dtype=bool))
    correlaciones_filtradas = matriz_correlacion.mask(mask)
    
    # Convertir a formato largo y ordenar por valor absoluto
    correlaciones_stack = correlaciones_filtradas.stack().reset_index()
    correlaciones_stack.columns = ['Variable_1', 'Variable_2', 'Correlacion']
    correlaciones_stack = correlaciones_stack.dropna()
    correlaciones_stack['Correlacion_Abs'] = correlaciones_stack['Correlacion'].abs()
    correlaciones_ordenadas = correlaciones_stack.sort_values('Correlacion_Abs', ascending=False)
    
    # Mostrar las 5 correlaciones más altas
    print("\n📊 Top 5 correlaciones más altas:")
    top_5 = correlaciones_ordenadas.head(5)[['Variable_1', 'Variable_2', 'Correlacion']]
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"   {i}. {row['Variable_1']} ↔ {row['Variable_2']}: {row['Correlacion']:.2f}")
    
    # Guardar archivos si se solicita
    if guardar_archivo:
        # Matriz completa
        archivo_correlaciones = "matriz_correlaciones.csv"
        matriz_correlacion.to_csv(archivo_correlaciones)
        print(f"\n💾 Matriz de correlaciones guardada en: {archivo_correlaciones}")
        
        # Top correlaciones con 2 decimales
        archivo_top_correlaciones = "top_correlaciones.csv"
        correlaciones_para_guardar = correlaciones_ordenadas[['Variable_1', 'Variable_2', 'Correlacion']].copy()
        correlaciones_para_guardar['Correlacion'] = correlaciones_para_guardar['Correlacion'].round(2)
        correlaciones_para_guardar.to_csv(archivo_top_correlaciones, index=False)
        print(f"💾 Top correlaciones guardadas en: {archivo_top_correlaciones}")
    
    # Generar mapa de calor si se solicita
    if mostrar_heatmap:
        plt.figure(figsize=(12, 10))
        sns.heatmap(matriz_correlacion, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={"shrink": .8})
        plt.title('Mapa de Calor - Correlaciones de Pearson', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('heatmap_correlaciones.png', dpi=300, bbox_inches='tight')
        print(f"🎨 Mapa de calor guardado en: heatmap_correlaciones.png")
        plt.show()
    
    return matriz_correlacion


# ================================================================================================
# FUNCIÓN PRINCIPAL
# ================================================================================================

def main():
    """
    Función principal que ejecuta el análisis de correlaciones y preprocesamiento de datos.
    
    Workflow:
        1. Carga de datos desde archivo CSV
        2. Exploración inicial del dataset
        3. Creación de variables binarias para fallos
        4. Imputación de valores faltantes
        5. Análisis de correlaciones de Pearson
        6. Generación de dataset procesado para modelado
        
    Environment Variables:
        rute (str): Ruta al archivo CSV con los datos (definida en .env)
        
    Files Generated:
        - matriz_correlaciones.csv: Matriz completa de correlaciones
        - top_correlaciones.csv: Correlaciones ordenadas por valor absoluto
        - heatmap_correlaciones.png: Visualización de correlaciones
        - Dataset_Talento_Procesado.csv: Dataset listo para machine learning
        
    Note:
        Para ejecutar modelos de árbol de decisión, usar: python decision_tree.py
        
    Raises:
        FileNotFoundError: Si no se encuentra el archivo de datos
        KeyError: Si faltan variables de entorno requeridas
    """
    print("🚀 Iniciando análisis de datos - Challenge Ingelean")
    print("=" * 60)
    
    # Cargar configuración y datos
    load_dotenv()
    processor = DataProcessor()
    rute = os.getenv("rute")
    
    if not rute:
        raise KeyError("❌ Variable de entorno 'rute' no encontrada. Verifica tu archivo .env")
    
    df = processor.load_data(rute)
    
    # 1. EXPLORACIÓN INICIAL
    print("\n📋 INFORMACIÓN BÁSICA DEL DATASET")
    print("-" * 40)
    print(f"✅ Dataset cargado exitosamente!")
    print(f"📊 Forma del dataset: {df.shape}")
    print(f"🏷️  Columnas ({len(df.columns)}): {list(df.columns)}")
    print(f"\n📄 Primeras 5 filas:")
    print(df.head())

    # 2. PREPROCESAMIENTO
    print(f"\n🔧 PREPROCESAMIENTO DE DATOS")
    print("-" * 40)
    
    # Crear columna de fallos binarios ANTES de la imputación
    df_con_fallos = crear_fallos_binarios(df, 'tipo_fallo')

    # Aplicar imputación a datos numéricos
    df_imputado = imputacion_simple(df_con_fallos, estrategia='media')
    
    # Mostrar información de valores nulos
    print(f"\n📊 Valores nulos por columna (antes de imputación):")
    print(df.isnull().sum())

    print(f"\n📊 Valores nulos por columna (después de imputación):")
    print(df_imputado.isnull().sum())

    # 3. ANÁLISIS DE CORRELACIONES
    print(f"\n🔍 ANÁLISIS DE CORRELACIONES")
    print("-" * 40)
    matriz_correlacion = crear_tabla_correlaciones(df_imputado, 
                                                  guardar_archivo=True,
                                                  mostrar_heatmap=True)

    # 4. EXPORTACIÓN DEL DATASET PROCESADO (SOLO VARIABLES NUMÉRICAS)
    print(f"\n💾 EXPORTACIÓN DE DATOS PROCESADOS")
    print("-" * 40)
    
    # Identificar variables categóricas a eliminar
    variables_categoricas = [
        'timestamp', 'turno', 'operador_id', 'maquina_id', 'producto_id',
        'fallo_detectado', 'tipo_fallo', 'observaciones'
    ]
    
    # Identificar variables numéricas a mantener
    variables_numericas = [
        'temperatura', 'vibracion', 'humedad', 'tiempo_ciclo',
        'cantidad_producida', 'unidades_defectuosas', 'eficiencia_porcentual',
        'consumo_energia', 'paradas_programadas', 'paradas_imprevistas',
        'fallos_binarios'
    ]
    
    # Filtrar solo las columnas que existen en el dataset
    variables_numericas_existentes = [col for col in variables_numericas if col in df_imputado.columns]
    variables_categoricas_existentes = [col for col in variables_categoricas if col in df_imputado.columns]
    
    print(f"🗂️  Filtrado de variables:")
    print(f"   • Variables categóricas eliminadas ({len(variables_categoricas_existentes)}): {variables_categoricas_existentes}")
    print(f"   • Variables numéricas mantenidas ({len(variables_numericas_existentes)}): {variables_numericas_existentes}")
    
    # Crear dataset solo con variables numéricas
    df_solo_numericas = df_imputado[variables_numericas_existentes].copy()
    
    # Exportar el dataset procesado (solo numéricas)
    archivo_procesado = "Dataset_Talento_Procesado.csv"
    df_solo_numericas.to_csv(archivo_procesado, index=False)
    
    print(f"\n✅ Dataset procesado exportado: {archivo_procesado}")
    print(f"📊 Cambios aplicados:")
    print(f"   • Variables categóricas eliminadas")
    print(f"   • Columna 'fallos_binarios' agregada")
    print(f"   • Valores faltantes imputados (estrategia: media)")
    print(f"   • {df_solo_numericas.shape[0]} registros × {df_solo_numericas.shape[1]} columnas")
    print(f"🎯 Dataset optimizado para machine learning (solo variables numéricas)")

    # 5. FINALIZACIÓN
    print(f"\n✅ ANÁLISIS DE CORRELACIONES COMPLETADO")
    print("=" * 60)
    print("📁 Archivos generados:")
    print("   • matriz_correlaciones.csv")
    print("   • top_correlaciones.csv") 
    print("   • heatmap_correlaciones.png")
    print("   • Dataset_Talento_Procesado.csv")
    print(f"\n🎯 Dataset procesado listo para modelado!")
    print(f"\n💡 Para ejecutar modelos de árbol de decisión:")
    print(f"   python decision_tree.py")

    # Opcional: Generar reporte de perfilado
    # print("\n📊 Generando reporte de perfilado...")
    # profile = ProfileReport(df, title="Reporte de Perfilado", explorative=True)
    # profile.to_file("reporte_perfilado.html")
    # print("✅ Reporte generado exitosamente: reporte_perfilado.html")


if __name__ == "__main__":
    main()
