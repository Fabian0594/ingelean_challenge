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

def crear_fallos_binarios(df, columna_fallo='tipo_fallo'):
    """
    Crea una columna binaria de fallos:
    - 0 donde hay NaN (sin fallo)
    - 1 donde no hay NaN (con fallo)
    
    Parámetros:
    df -- DataFrame original
    columna_fallo -- nombre de la columna a evaluar para crear la binaria
    
    Retorna:
    DataFrame con la nueva columna 'fallos_binarios'
    """
    df_con_binarios = df.copy()
    
    # Crear columna binaria: 1 si NO es NaN (hay fallo), 0 si es NaN (sin fallo)
    df_con_binarios['fallos_binarios'] = (~df_con_binarios[columna_fallo].isna()).astype(int)
    
    print(f"\nColumna 'fallos_binarios' creada:")
    print(f"- Sin fallo (0): {(df_con_binarios['fallos_binarios'] == 0).sum()} registros")
    print(f"- Con fallo (1): {(df_con_binarios['fallos_binarios'] == 1).sum()} registros")
    
    return df_con_binarios


def imputacion_simple(df, estrategia='media'):
    """Version simplificada para imputacion basica"""
    # Crear una copia del DataFrame original
    df_imputado = df.copy()
    
    # Identificar columnas numéricas
    columnas_numericas = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    
    # Aplicar imputación solo a columnas numéricas
    if estrategia == 'media':
        df_imputado[columnas_numericas] = df_imputado[columnas_numericas].fillna(df_imputado[columnas_numericas].mean())
    else:
        df_imputado[columnas_numericas] = df_imputado[columnas_numericas].fillna(df_imputado[columnas_numericas].median())
    
    return df_imputado


def pearson_correlation(df, col1, col2):
    """
    Calcula la correlación de Pearson entre dos columnas de un DataFrame.
    
    Parámetros:
    df : DataFrame
        DataFrame de pandas
    col1, col2 : str
        Nombres de las columnas a correlacionar
        
    Retorna:
    float
        Coeficiente de correlación de Pearson
    """
    return df[[col1, col2]].corr().iloc[0, 1]

def crear_tabla_correlaciones(df, guardar_archivo=True, mostrar_heatmap=True):
    """
    Crea una tabla de correlaciones de Pearson para todas las variables numéricas.
    
    Parámetros:
    df : DataFrame
        DataFrame de pandas
    guardar_archivo : bool
        Si True, guarda la tabla en un archivo CSV
    mostrar_heatmap : bool
        Si True, muestra un mapa de calor de las correlaciones
        
    Retorna:
    DataFrame
        Matriz de correlaciones de Pearson
    """
    # Seleccionar solo columnas numéricas
    columnas_numericas = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    
    if len(columnas_numericas) == 0:
        print("No se encontraron columnas numéricas para calcular correlaciones.")
        return None
    
    print(f"\nCalculando correlaciones de Pearson para {len(columnas_numericas)} variables numéricas...")
    
    # Calcular matriz de correlación
    matriz_correlacion = df[columnas_numericas].corr(method='pearson')
    
    # Mostrar información básica
    print("✓ Matriz de correlaciones calculada exitosamente")
    
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
    print("\nTop 5 correlaciones más altas:")
    top_5 = correlaciones_ordenadas.head(5)[['Variable_1', 'Variable_2', 'Correlacion']]
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"{i}. {row['Variable_1']} ↔ {row['Variable_2']}: {row['Correlacion']:.2f}")
    
    # Guardar en archivo CSV si se solicita
    if guardar_archivo:
        archivo_correlaciones = "matriz_correlaciones.csv"
        matriz_correlacion.to_csv(archivo_correlaciones)
        print(f"\n✓ Matriz de correlaciones guardada en: {archivo_correlaciones}")
        
        archivo_top_correlaciones = "top_correlaciones.csv"
        correlaciones_para_guardar = correlaciones_ordenadas[['Variable_1', 'Variable_2', 'Correlacion']].copy()
        correlaciones_para_guardar['Correlacion'] = correlaciones_para_guardar['Correlacion'].round(2)
        correlaciones_para_guardar.to_csv(archivo_top_correlaciones, index=False)
        print(f"✓ Top correlaciones guardadas en: {archivo_top_correlaciones}")
    
    # Mostrar mapa de calor si se solicita
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
        print(f"✓ Mapa de calor guardado en: heatmap_correlaciones.png")
        plt.show()
    
    return matriz_correlacion

def main():
    load_dotenv()
    processor = DataProcessor()
    rute = os.getenv("rute")
    df = processor.load_data(rute)
    
    # Mostrar información básica del dataset
    print("Dataset cargado exitosamente!")
    print(f"Forma del dataset: {df.shape}")
    print(f"Columnas: {list(df.columns)}")
    print("\nPrimeras 5 filas:")
    print(df.head())

    # Crear columna de fallos binarios ANTES de la imputación
    df_con_fallos = crear_fallos_binarios(df, 'tipo_fallo')

    # Aplicar imputación a datos numéricos
    df_imputado = imputacion_simple(df_con_fallos)
    
    # Mostrar información de valores nulos después de la imputación
    print("\nValores nulos por columna (antes de imputación):")
    print(df.isnull().sum())

    # Mostrar información de valores nulos después de la imputación
    print("\nValores nulos por columna (después de imputación):")
    print(df_imputado.isnull().sum())

    # Crear tabla de correlaciones de Pearson
    matriz_correlacion = crear_tabla_correlaciones(df_imputado, 
                                                  guardar_archivo=True,
                                                  mostrar_heatmap=True)

    # Generar reporte de perfilado
    # print("\nGenerando reporte de perfilado...")
    # profile = ProfileReport(df, title="Reporte de Perfilado", explorative=True)
    # profile.to_file("reporte_perfilado.html")
    # print("¡Reporte generado exitosamente: reporte_perfilado.html")


if __name__ == "__main__":
    main()