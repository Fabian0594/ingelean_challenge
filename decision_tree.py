"""
Modelo de Árbol de Decisión para Análisis de Eficiencia - Challenge Ingelean
=========================================================================

Este módulo implementa un modelo de árbol de decisión para predecir la eficiencia
porcentual usando el dataset procesado de manufactura.

Autor: Fabián Rodriguez
Fecha: 2024
"""

import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

class DecisionTreeModel:
    """
    Modelo de Árbol de Decisión para predicción de eficiencia porcentual.
    
    Esta clase implementa un modelo de árbol de decisión completo con preprocesamiento
    automático, entrenamiento, evaluación y visualización de resultados.
    
    Attributes:
        df (pd.DataFrame): Dataset con los datos a procesar
        target_column (str): Nombre de la columna objetivo
        model (DecisionTreeRegressor): Modelo de sklearn entrenado
        results (dict): Diccionario con métricas y resultados del modelo
        preprocessors (dict): Diccionario con objetos de preprocesamiento
    """
    
    def __init__(self, data_path=None, df=None, target_column='eficiencia_porcentual'):
        """
        Inicializa el modelo de árbol de decisión.
        
        Args:
            data_path (str, optional): Ruta al archivo de datos (Excel o CSV)
            df (pd.DataFrame, optional): DataFrame directamente cargado
            target_column (str, optional): Nombre de la columna objetivo.
                                         Por defecto 'eficiencia_porcentual'
        
        Raises:
            ValueError: Si no se proporciona ni data_path ni df
            
        Example:
            >>> # Usando DataFrame directamente
            >>> model = DecisionTreeModel(df=mi_dataframe)
            >>> 
            >>> # Usando archivo
            >>> model = DecisionTreeModel(data_path="datos.csv")
        """
        if df is not None:
            self.df = df.copy()
        elif data_path:
            self.load_data(data_path)
        else:
            raise ValueError("Debe proporcionar data_path o df")
            
        self.target_column = target_column
        self.model = None
        self.results = {}
        self.preprocessors = {}
        
    def load_data(self, data_path):
        """
        Carga los datos desde un archivo Excel o CSV.
        
        Args:
            data_path (str): Ruta al archivo de datos
            
        Raises:
            Exception: Si hay problemas al cargar el archivo
            
        Note:
            Detecta automáticamente el formato del archivo basado en la extensión.
        """
        try:
            if data_path.endswith('.xlsx'):
                self.df = pd.read_excel(data_path)
            else:
                self.df = pd.read_csv(data_path)
            print(f"Datos cargados exitosamente. Forma: {self.df.shape}")
        except Exception as e:
            print(f"Error al cargar los datos: {e}")
            raise

    def preprocess_data(self):
        """
        Preprocesa los datos para el entrenamiento del modelo de árbol de decisión.
        
        El preprocesamiento incluye:
        1. Limpieza de valores extremos y faltantes en la variable objetivo
        2. Identificación de columnas categóricas y numéricas
        3. Codificación de variables categóricas usando LabelEncoder
        4. Imputación de valores faltantes en variables numéricas (media)
        5. Preparación del dataset final para entrenamiento
        
        Note:
            - Los valores extremos (>1e16) se eliminan para evitar problemas numéricos
            - Las columnas categóricas se codifican numéricamente
            - Los valores faltantes se imputan con la media de cada columna
            - Se actualiza self.df con los datos preprocesados
            
        Raises:
            KeyError: Si las columnas especificadas no existen en el dataset
        """
        # Limpieza básica
        self.df = self.df[self.df[self.target_column].notna()]
        self.df = self.df[self.df[self.target_column] < 1e16]  # Eliminar valores extremos
        
        # Columnas categóricas y numéricas
        self.categorical_cols = ['turno', 'operador_id', 'maquina_id', 'producto_id', 'fallo_detectado', 'tipo_fallo']
        self.numeric_cols = ['temperatura', 'vibracion', 'humedad', 'tiempo_ciclo', 
                           'cantidad_producida', 'unidades_defectuosas', 
                           'consumo_energia', 'paradas_programadas', 'paradas_imprevistas', 'fallos_binarios']
        
        # Asegurar que las columnas existan
        self.categorical_cols = [col for col in self.categorical_cols if col in self.df.columns]
        self.numeric_cols = [col for col in self.numeric_cols if col in self.df.columns]
        
        # 1. Codificar variables categóricas
        self.label_encoders = {}
        for col in self.categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
        
        # 2. Imputar valores faltantes
        self.imputer = SimpleImputer(strategy='median')
        self.df[self.numeric_cols] = self.imputer.fit_transform(self.df[self.numeric_cols])
        
        # 3. Normalización (guardamos parámetros para nuevos datos)
        self.numeric_means = self.df[self.numeric_cols].mean()
        self.numeric_stds = self.df[self.numeric_cols].std()
        self.df[self.numeric_cols] = (self.df[self.numeric_cols] - self.numeric_means) / self.numeric_stds
        
        # Guardar preprocesadores
        self.preprocessors = {
            'label_encoders': self.label_encoders,
            'imputer': self.imputer,
            'numeric_means': self.numeric_means,
            'numeric_stds': self.numeric_stds
        }

    def train_model(self, test_size=0.2, random_state=42, max_depth=5):
        """
        Entrena el modelo de árbol de decisión y evalúa su rendimiento.
        
        Args:
            test_size (float, optional): Proporción del dataset para conjunto de prueba.
                                       Por defecto 0.2 (20%)
            random_state (int, optional): Semilla para reproducibilidad de resultados.
                                        Por defecto 42
            max_depth (int, optional): Profundidad máxima del árbol para evitar overfitting.
                                     Por defecto 5
        
        Returns:
            dict: Diccionario con métricas de evaluación y resultados del modelo.
                 Incluye: 'metrics', 'feature_importance', 'predictions'
        
        Note:
            - Divide automáticamente los datos en entrenamiento y prueba
            - Calcula métricas: MSE, RMSE, MAE, R²
            - Genera tabla de importancia de características
            - Almacena resultados en self.results
            
        Example:
            >>> model = DecisionTreeModel(df=data)
            >>> model.preprocess_data()
            >>> results = model.train_model(test_size=0.3, max_depth=7)
            >>> print(f"R² Score: {results['metrics']['r2']:.4f}")
        """
        # Preparar datos
        X = self.df[self.categorical_cols + self.numeric_cols]
        y = self.df[self.target_column]
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Crear y entrenar modelo
        self.model = DecisionTreeRegressor(
            max_depth=max_depth, 
            random_state=random_state
        )
        self.model.fit(X_train, y_train)
        
        # Evaluar
        y_pred = self.model.predict(X_test)
        
        # Calcular métricas
        self.results = {
            'metrics': {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            },
            'feature_importance': self.get_feature_importance(),
            'test_predictions': pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred
            })
        }
        
        return self.results

    def get_feature_importance(self):
        """Obtiene la importancia de las características"""
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado aún")
            
        importances = self.model.feature_importances_
        features = self.categorical_cols + self.numeric_cols
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return feature_importance

    def plot_tree(self, max_depth=3, figsize=(20, 10)):
        """Visualiza el árbol de decisión (parcial para evitar sobrecarga)"""
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado aún")
            
        plt.figure(figsize=figsize)
        plot_tree(
            self.model, 
            feature_names=self.categorical_cols + self.numeric_cols,
            filled=True, 
            rounded=True,
            max_depth=max_depth,
            fontsize=10
        )
        plt.title("Árbol de Decisión (primeros niveles)", pad=20)
        plt.tight_layout()
        plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_importance(self, top_n=10):
        """Grafica la importancia de las características"""
        importance_df = self.get_feature_importance()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x='Importance', 
            y='Feature', 
            data=importance_df.head(top_n),
            palette='viridis'
        )
        plt.title(f'Top {top_n} Características Más Importantes', pad=20)
        plt.xlabel('Importancia')
        plt.ylabel('Característica')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

    def print_results(self):
        """Imprime los resultados del modelo"""
        if not self.results:
            print("No hay resultados para mostrar. Entrene el modelo primero.")
            return
        
        print("\n" + "="*50)
        print("RESULTADOS DEL ÁRBOL DE DECISIÓN")
        print("="*50)
        
        # Métricas
        print("\nMétricas de rendimiento:")
        print(f"- Error Cuadrático Medio (MSE): {self.results['metrics']['mse']:.4f}")
        print(f"- Raíz del Error Cuadrático Medio (RMSE): {self.results['metrics']['rmse']:.4f}")
        print(f"- Error Absoluto Medio (MAE): {self.results['metrics']['mae']:.4f}")
        print(f"- Coeficiente R²: {self.results['metrics']['r2']:.4f}")
        
        # Importancia de características
        print("\nImportancia de las características:")
        print(self.results['feature_importance'].to_string(index=False))
        
        print("\n" + "="*50)

def main():
    """
    Función principal que ejecuta el modelo de árbol de decisión usando el dataset procesado.
    
    Workflow:
        1. Carga del dataset procesado (Dataset_Talento_Procesado.csv)
        2. Verificación de datos
        3. Entrenamiento del modelo de árbol de decisión
        4. Evaluación y visualización de resultados
        
    Files Used:
        - Dataset_Talento_Procesado.csv: Dataset con preprocesamiento aplicado
        
    Files Generated:
        - feature_importance.png: Gráfico de importancia de características
        - decision_tree.png: Visualización del árbol de decisión
    """
    print("🌳 Iniciando Modelo de Árbol de Decisión - Challenge Ingelean")
    print("=" * 65)
    
    # 1. CARGAR DATASET PROCESADO
    print("\n📁 CARGANDO DATASET PROCESADO")
    print("-" * 40)
    
    dataset_path = "Dataset_Talento_Procesado.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: No se encontró el archivo {dataset_path}")
        print("💡 Ejecuta primero main.py para generar el dataset procesado")
        return
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset cargado exitosamente: {dataset_path}")
        print(f"📊 Forma del dataset: {df.shape}")
        print(f"🔍 Columnas disponibles: {list(df.columns)}")
        
        # Verificar que la columna objetivo existe
        target_column = 'eficiencia_porcentual'
        if target_column not in df.columns:
            print(f"❌ Error: No se encontró la columna objetivo '{target_column}'")
            return
            
        print(f"🎯 Variable objetivo: {target_column}")
        
    except Exception as e:
        print(f"❌ Error al cargar el dataset: {str(e)}")
        return

    # 2. INFORMACIÓN DEL DATASET
    print(f"\n📋 INFORMACIÓN DEL DATASET PROCESADO")
    print("-" * 40)
    print(f"✅ Valores nulos eliminados mediante imputación")
    print(f"✅ Variable 'fallos_binarios' agregada")
    print(f"📊 Valores nulos restantes:")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0] if nulls.sum() > 0 else "   Ninguno - Dataset completamente limpio")

    # 3. MODELADO CON ÁRBOL DE DECISIÓN
    print(f"\n🌳 ENTRENAMIENTO DEL MODELO")
    print("-" * 40)
    
    # Crear instancia del modelo
    dt_model = DecisionTreeModel(df=df, target_column=target_column)
    
    # Preprocesar datos
    print("🔧 Preprocesando datos para el modelo...")
    dt_model.preprocess_data()
    
    # Entrenar modelo
    print("⚙️ Entrenando árbol de decisión...")
    results = dt_model.train_model(test_size=0.2, random_state=42, max_depth=5)
    
    # 4. RESULTADOS Y VISUALIZACIONES
    print(f"\n📊 RESULTADOS DEL MODELO")
    print("-" * 40)
    
    # Mostrar resultados
    dt_model.print_results()
    
    # Generar visualizaciones
    print(f"\n🎨 GENERANDO VISUALIZACIONES")
    print("-" * 40)
    dt_model.plot_feature_importance(top_n=10)
    dt_model.plot_tree(max_depth=3)
    
    # 5. FINALIZACIÓN
    print(f"\n✅ MODELADO COMPLETADO")
    print("=" * 65)
    print("📁 Archivos generados:")
    print("   • feature_importance.png")
    print("   • decision_tree.png")
    print(f"🎯 Modelo de árbol de decisión listo!")
    print(f"📈 R² Score: {results['metrics']['r2']:.4f}")
    print(f"📉 RMSE: {results['metrics']['rmse']:.4f}")

if __name__ == "__main__":
    main()