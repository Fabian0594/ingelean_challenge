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
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

def aplicar_arbol_decision(X, y, test_size=0.3, random_state=42, max_depth=None, criterion='gini'):
    """
    Aplica un modelo de Árbol de Decisión y devuelve el modelo entrenado y las métricas de evaluación.
    
    Parámetros:
    -----------
    X : array-like
        Variables predictoras (features)
    y : array-like
        Variable objetivo (target)
    test_size : float, opcional (default=0.3)
        Proporción del conjunto de prueba
    random_state : int, opcional (default=42)
        Semilla para reproducibilidad
    max_depth : int or None, opcional (default=None)
        Profundidad máxima del árbol. None significa que se expande hasta que todas las hojas sean puras.
    criterion : str, opcional (default='gini')
        Función para medir la calidad de una división ('gini' o 'entropy')
    
    Retorna:
    --------
    model : DecisionTreeClassifier
        Modelo entrenado
    metrics : dict
        Diccionario con las métricas de evaluación
    """
    
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Crear y entrenar el modelo
    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Predecir en el conjunto de prueba
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }
    
    return model, metrics

def mostrar_metricas_clasificacion(metrics, feature_names=None, model=None):
    """
    Muestra las métricas de clasificación de manera legible y organizada.
    
    Parámetros:
    -----------
    metrics : dict
        Diccionario con métricas retornado por aplicar_arbol_decision
    feature_names : list, opcional
        Lista con nombres de las características para mostrar importancia
    model : DecisionTreeClassifier, opcional
        Modelo entrenado para mostrar importancia de características
    """
    print("\n" + "="*60)
    print("🎯 RESULTADOS DE CLASIFICACIÓN DE FALLOS")
    print("="*60)
    
    # 1. PRECISIÓN GENERAL
    print(f"\n📈 MÉTRICAS GENERALES")
    print("-" * 40)
    accuracy = metrics['accuracy']
    print(f"• Precisión (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Interpretación de la precisión
    if accuracy >= 0.95:
        acc_interpretation = "Excelente precisión 🎯"
    elif accuracy >= 0.90:
        acc_interpretation = "Muy buena precisión 👍"
    elif accuracy >= 0.80:
        acc_interpretation = "Buena precisión ✅"
    elif accuracy >= 0.70:
        acc_interpretation = "Precisión aceptable 🤔"
    else:
        acc_interpretation = "Precisión baja - revisar modelo ⚠️"
    
    print(f"• Interpretación: {acc_interpretation}")
    
    # 2. MATRIZ DE CONFUSIÓN
    print(f"\n🔍 MATRIZ DE CONFUSIÓN")
    print("-" * 40)
    conf_matrix = metrics['confusion_matrix']
    
    # Etiquetas para la matriz
    print("Matriz de Confusión:")
    print("                 Predicción")
    print("                Sin Fallo  Con Fallo")
    print(f"Real Sin Fallo      {conf_matrix[0,0]:4d}      {conf_matrix[0,1]:4d}")
    print(f"Real Con Fallo      {conf_matrix[1,0]:4d}      {conf_matrix[1,1]:4d}")
    
    # Calcular métricas adicionales
    tn, fp, fn, tp = conf_matrix.ravel()
    
    print(f"\nInterpretación:")
    print(f"• Verdaderos Negativos (TN): {tn} - Sin fallo predicho correctamente")
    print(f"• Falsos Positivos (FP): {fp} - Fallos predichos incorrectamente")
    print(f"• Falsos Negativos (FN): {fn} - Fallos no detectados")
    print(f"• Verdaderos Positivos (TP): {tp} - Fallos detectados correctamente")
    
    # 3. MÉTRICAS POR CLASE
    print(f"\n📊 MÉTRICAS DETALLADAS POR CLASE")
    print("-" * 40)
    report = metrics['classification_report']
    
    # Clase 0 (Sin Fallo)
    class_0 = report['0']
    print(f"Sin Fallo (Clase 0):")
    print(f"  • Precisión:  {class_0['precision']:.4f}")
    print(f"  • Recall:     {class_0['recall']:.4f}")
    print(f"  • F1-Score:   {class_0['f1-score']:.4f}")
    print(f"  • Soporte:    {class_0['support']} muestras")
    
    # Clase 1 (Con Fallo)
    class_1 = report['1']
    print(f"\nCon Fallo (Clase 1):")
    print(f"  • Precisión:  {class_1['precision']:.4f}")
    print(f"  • Recall:     {class_1['recall']:.4f}")
    print(f"  • F1-Score:   {class_1['f1-score']:.4f}")
    print(f"  • Soporte:    {class_1['support']} muestras")
    
    # Métricas macro y weighted
    print(f"\nPromedios:")
    print(f"  • Macro avg F1:    {report['macro avg']['f1-score']:.4f}")
    print(f"  • Weighted avg F1: {report['weighted avg']['f1-score']:.4f}")
    
    # 4. IMPORTANCIA DE CARACTERÍSTICAS (si está disponible)
    if model is not None and feature_names is not None:
        print(f"\n🎯 IMPORTANCIA DE CARACTERÍSTICAS")
        print("-" * 40)
        
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("Top 10 características más importantes para predecir fallos:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            feature_name = row['Feature']
            importance = row['Importance']
            
            # Crear barra visual
            bar_length = int(importance * 30)
            bar = "█" * min(bar_length, 15) + "░" * max(0, 8 - bar_length)
            
            print(f"{i:2d}. {feature_name:<22} {bar} {importance:.4f}")
    
    # 5. RECOMENDACIONES
    print(f"\n💡 RECOMENDACIONES")
    print("-" * 40)
    
    # Análisis de falsos positivos y negativos
    if fp > fn:
        print("• Hay más falsos positivos que negativos")
        print("• El modelo tiende a 'sobrediagnosticar' fallos")
        print("• Considerar ajustar el umbral de decisión")
    elif fn > fp:
        print("• Hay más falsos negativos que positivos")
        print("• El modelo podría estar perdiendo fallos reales")
        print("• Importante en contexto industrial - revisar parámetros")
    else:
        print("• Balance equilibrado entre falsos positivos y negativos")
    
    # Análisis del recall para fallos
    if class_1['recall'] < 0.8:
        print("• Recall bajo para detectar fallos - crítico en manufactura")
        print("• Considerar ajustar max_depth o usar criterion='entropy'")
    else:
        print("• Buen recall para detección de fallos")
    
    print("\n" + "="*60)

def clasificar_fallos_con_arbol(data_path="Dataset_Talento_Procesado.csv", test_size=0.3, max_depth=5, criterion='gini'):
    """
    Aplica clasificación de fallos usando la función aplicar_arbol_decision.
    
    Parámetros:
    -----------
    data_path : str
        Ruta al dataset procesado
    test_size : float
        Proporción del conjunto de prueba
    max_depth : int
        Profundidad máxima del árbol
    criterion : str
        Criterio de división ('gini' o 'entropy')
    
    Retorna:
    --------
    model : DecisionTreeClassifier
        Modelo entrenado
    metrics : dict
        Métricas de evaluación
    """
    print("🎯 Iniciando Clasificación de Fallos con Árbol de Decisión")
    print("=" * 65)
    
    # 1. CARGAR DATOS
    print(f"\n📁 CARGANDO DATASET: {data_path}")
    print("-" * 40)
    
    if not os.path.exists(data_path):
        print(f"❌ Error: No se encontró {data_path}")
        print("💡 Ejecuta main.py primero para generar el dataset procesado")
        return None, None
    
    df = pd.read_csv(data_path)
    print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # 2. VERIFICAR VARIABLE OBJETIVO
    if 'fallos_binarios' not in df.columns:
        print("❌ Error: No se encontró la columna 'fallos_binarios'")
        print("💡 Asegúrate de que main.py haya creado esta columna")
        return None, None
    
    # 3. PREPARAR DATOS
    print(f"\n🔧 PREPARANDO DATOS PARA CLASIFICACIÓN")
    print("-" * 40)
    
    # Variables predictoras (todas excepto la objetivo)
    feature_cols = [col for col in df.columns if col != 'fallos_binarios']
    X = df[feature_cols]
    y = df['fallos_binarios']
    
    print(f"• Features: {len(feature_cols)} variables")
    print(f"• Target: fallos_binarios")
    print(f"• Distribución de clases:")
    print(f"  - Sin fallo (0): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
    print(f"  - Con fallo (1): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
    
    # 4. APLICAR MODELO USANDO LA FUNCIÓN DEL USUARIO
    print(f"\n🌳 ENTRENANDO MODELO DE CLASIFICACIÓN")
    print("-" * 40)
    print(f"• Conjunto de prueba: {test_size*100:.0f}%")
    print(f"• Profundidad máxima: {max_depth}")
    print(f"• Criterio: {criterion}")
    
    model, metrics = aplicar_arbol_decision(
        X=X, 
        y=y, 
        test_size=test_size, 
        random_state=42, 
        max_depth=max_depth, 
        criterion=criterion
    )
    
    # 5. MOSTRAR RESULTADOS
    mostrar_metricas_clasificacion(metrics, feature_names=feature_cols, model=model)
    
    # 6. GENERAR VISUALIZACIONES
    print(f"\n🎨 GENERANDO VISUALIZACIONES")
    print("-" * 40)
    
    # Gráfico de importancia de características
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
    plt.title('Importancia de Características para Clasificación de Fallos', fontsize=14, pad=20)
    plt.xlabel('Importancia')
    plt.ylabel('Característica')
    plt.tight_layout()
    plt.savefig('feature_importance_clasificacion.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualización de la matriz de confusión
    plt.figure(figsize=(8, 6))
    conf_matrix = metrics['confusion_matrix']
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Sin Fallo', 'Con Fallo'],
                yticklabels=['Sin Fallo', 'Con Fallo'])
    plt.title('Matriz de Confusión - Clasificación de Fallos', fontsize=14, pad=20)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualización parcial del árbol
    plt.figure(figsize=(20, 12))
    plot_tree(model, 
              feature_names=feature_cols,
              class_names=['Sin Fallo', 'Con Fallo'],
              filled=True, 
              rounded=True,
              max_depth=3,  # Solo primeros niveles
              fontsize=10)
    plt.title('Árbol de Decisión para Clasificación de Fallos (primeros 3 niveles)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('decision_tree_clasificacion.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Visualizaciones guardadas:")
    print(f"   • feature_importance_clasificacion.png")
    print(f"   • confusion_matrix.png")
    print(f"   • decision_tree_clasificacion.png")
    
    print(f"\n✅ CLASIFICACIÓN COMPLETADA")
    print("=" * 65)
    
    return model, metrics

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
        
        Como el dataset procesado ya contiene solo variables numéricas limpias,
        el preprocesamiento es mínimo:
        1. Limpieza básica de la variable objetivo
        2. Identificación de variables predictoras (todas numéricas)
        3. Verificación de integridad de datos
        
        Note:
            - El dataset de entrada ya está preprocesado (solo variables numéricas)
            - No se requiere codificación de variables categóricas
            - No se requiere imputación (ya aplicada en main.py)
            - No se aplica normalización (innecesaria para árboles de decisión)
            
        Raises:
            ValueError: Si la variable objetivo no existe o tiene problemas
        """
        print("🔧 Preprocesamiento simplificado (dataset solo con variables numéricas)...")
        
        # 1. LIMPIEZA BÁSICA DE VARIABLE OBJETIVO
        initial_shape = self.df.shape
        
        # Verificar que la variable objetivo existe
        if self.target_column not in self.df.columns:
            raise ValueError(f"Variable objetivo '{self.target_column}' no encontrada")
        
        # Limpiar valores extremos y faltantes en variable objetivo
        self.df = self.df[self.df[self.target_column].notna()]
        self.df = self.df[self.df[self.target_column] < 1e16]
        
        if self.df.shape[0] < initial_shape[0]:
            eliminados = initial_shape[0] - self.df.shape[0]
            print(f"   • Registros eliminados por problemas en variable objetivo: {eliminados}")
        
        # 2. IDENTIFICACIÓN DE VARIABLES PREDICTORAS
        # Todas las columnas excepto la variable objetivo son numéricas predictoras
        self.feature_cols = [col for col in self.df.columns if col != self.target_column]
        
        print(f"   • Variables predictoras: {len(self.feature_cols)}")
        print(f"   • Variable objetivo: {self.target_column}")
        
        # 3. VERIFICACIÓN DE INTEGRIDAD (por si acaso)
        missing_values = self.df[self.feature_cols].isnull().sum()
        if missing_values.sum() > 0:
            print(f"   ⚠️  Valores faltantes encontrados (esto no debería pasar): {missing_values.sum()}")
            # Aplicar imputación de emergencia
            self.imputer = SimpleImputer(strategy='median')
            self.df[self.feature_cols] = self.imputer.fit_transform(self.df[self.feature_cols])
            print("   • Imputación de emergencia aplicada")
        else:
            print("   ✅ Sin valores faltantes - Dataset listo para entrenamiento")
            self.imputer = None
        
        # 4. GUARDAR INFORMACIÓN SIMPLE
        self.preprocessors = {
            'imputer': self.imputer,
            'feature_cols': self.feature_cols
        }
        
        print(f"   ✅ Preprocesamiento completado: {self.df.shape[0]} registros × {len(self.feature_cols)} características")

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
        X = self.df[self.feature_cols]
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
        features = self.feature_cols
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
            feature_names=self.feature_cols,
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
            hue='Feature',
            palette='viridis',
            legend=False
        )
        plt.title(f'Top {top_n} Características Más Importantes', pad=20)
        plt.xlabel('Importancia')
        plt.ylabel('Característica')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

    def print_results(self):
        """
        Imprime los resultados del modelo con formato mejorado.
        
        Muestra métricas de rendimiento, importancia de características y
        análisis de los resultados del modelo entrenado.
        """
        if not self.results:
            print("❌ No hay resultados para mostrar. Entrene el modelo primero.")
            return
        
        print("\n" + "="*60)
        print("🌳 RESULTADOS DEL ÁRBOL DE DECISIÓN")
        print("="*60)
        
        # 1. INFORMACIÓN DEL DATASET
        print(f"\n📊 INFORMACIÓN DEL MODELO")
        print("-" * 40)
        print(f"• Variables predictoras: {len(self.feature_cols)} (todas numéricas)")
        print(f"• Variable objetivo: {self.target_column}")
        print(f"• Registros procesados: {self.df.shape[0]}")
        
        # 2. MÉTRICAS DE RENDIMIENTO
        print(f"\n📈 MÉTRICAS DE RENDIMIENTO")
        print("-" * 40)
        metrics = self.results['metrics']
        
        print(f"• Error Cuadrático Medio (MSE):     {metrics['mse']:.4f}")
        print(f"• Raíz del Error Cuadrático (RMSE): {metrics['rmse']:.4f}")
        print(f"• Error Absoluto Medio (MAE):       {metrics['mae']:.4f}")
        print(f"• Coeficiente R²:                   {metrics['r2']:.4f}")
        
        # Interpretación del R²
        if metrics['r2'] >= 0.8:
            r2_interpretation = "Excelente ajuste 🎯"
        elif metrics['r2'] >= 0.6:
            r2_interpretation = "Buen ajuste 👍"
        elif metrics['r2'] >= 0.4:
            r2_interpretation = "Ajuste moderado 🤔"
        elif metrics['r2'] >= 0.0:
            r2_interpretation = "Ajuste débil 😕"
        else:
            r2_interpretation = "Modelo no predictivo ❌"
        
        print(f"• Interpretación R²:                {r2_interpretation}")
        
        # 3. IMPORTANCIA DE CARACTERÍSTICAS
        print(f"\n🎯 IMPORTANCIA DE CARACTERÍSTICAS")
        print("-" * 40)
        importance_df = self.results['feature_importance']
        
        # Mostrar top 10 características
        top_features = importance_df.head(10)
        print("Top 10 características más importantes:")
        
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            feature_name = row['Feature']
            importance = row['Importance']
            
            # Crear barra visual simple
            bar_length = int(importance * 30)  # Escalar a 30 caracteres max
            bar = "█" * bar_length + "░" * (10 - bar_length) if bar_length < 10 else "█" * 10
            
            print(f"{i:2d}. {feature_name:<22} {bar} {importance:.4f}")
        
        # 4. RECOMENDACIONES
        print(f"\n💡 RECOMENDACIONES")
        print("-" * 40)
        if metrics['r2'] < 0.3:
            print("• El modelo tiene bajo poder predictivo")
            print("• Considerar feature engineering o modelos más complejos")
            print("• Revisar si hay variables importantes faltantes")
        elif metrics['r2'] < 0.6:
            print("• El modelo tiene capacidad predictiva moderada")
            print("• Considerar ensemble methods (Random Forest, XGBoost)")
        else:
            print("• El modelo tiene buen poder predictivo")
            print("• Considerar validación cruzada para confirmar estabilidad")
        
        # Análisis de características importantes
        top_feature = importance_df.iloc[0]
        if top_feature['Importance'] > 0.3:
            print(f"• La variable '{top_feature['Feature']}' es dominante")
            print("• Verificar si esto es esperado en el contexto del negocio")
        
        print("\n" + "="*60)

def main():
    """
    Función principal que ejecuta modelos de árbol de decisión usando el dataset procesado.
    
    Ofrece dos opciones:
    1. Regresión: Predecir eficiencia_porcentual (modelo original)
    2. Clasificación: Predecir fallos_binarios (usando función aplicar_arbol_decision)
    
    Workflow:
        1. Carga del dataset procesado (Dataset_Talento_Procesado.csv)
        2. Selección del tipo de modelo
        3. Entrenamiento y evaluación
        4. Visualización de resultados
        
    Files Used:
        - Dataset_Talento_Procesado.csv: Dataset con preprocesamiento aplicado
        
    Files Generated:
        - Para regresión: feature_importance.png, decision_tree.png
        - Para clasificación: feature_importance_clasificacion.png, confusion_matrix.png, decision_tree_clasificacion.png
    """
    print("🌳 Iniciando Modelos de Árbol de Decisión - Challenge Ingelean")
    print("=" * 70)
    
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

    # 3. SELECCIÓN DEL TIPO DE MODELO
    print(f"\n🎯 SELECCIÓN DEL TIPO DE MODELO")
    print("-" * 40)
    print("Opciones disponibles:")
    print("1. Regresión: Predecir eficiencia porcentual (modelo original)")
    print("2. Clasificación: Predecir fallos binarios (nueva función)")
    
    # Por defecto ejecutar ambos modelos para demostración completa
    print("\n🚀 Ejecutando ambos modelos para análisis completo...\n")
    
    # 4A. MODELO DE CLASIFICACIÓN DE FALLOS (NUEVO)
    print("="*70)
    print("🎯 MODELO 1: CLASIFICACIÓN DE FALLOS")
    print("="*70)
    
    model_clasificacion, metrics_clasificacion = clasificar_fallos_con_arbol(
        data_path=dataset_path,
        test_size=0.3,
        max_depth=5,
        criterion='gini'
    )
    
    # 4B. MODELO DE REGRESIÓN ORIGINAL 
    print("\n" + "="*70)
    print("📈 MODELO 2: REGRESIÓN DE EFICIENCIA")
    print("="*70)
    
    # Verificar que la columna objetivo existe para regresión
    target_column = 'eficiencia_porcentual'
    if target_column not in df.columns:
        print(f"❌ Error: No se encontró la columna objetivo '{target_column}'")
        print("💡 Continuando solo con clasificación...")
    else:
        print(f"🎯 Variable objetivo: {target_column}")
        
        # Crear instancia del modelo
        dt_model = DecisionTreeModel(df=df, target_column=target_column)
        
        # Preprocesar datos
        print("🔧 Preprocesando datos para el modelo...")
        dt_model.preprocess_data()
        
        # Entrenar modelo
        print("⚙️ Entrenando árbol de decisión...")
        results = dt_model.train_model(test_size=0.2, random_state=42, max_depth=5)
        
        # Mostrar resultados
        dt_model.print_results()
        
        # Generar visualizaciones
        #print(f"\n🎨 GENERANDO VISUALIZACIONES PARA REGRESIÓN")
        #print("-" * 40)
        #dt_model.plot_feature_importance(top_n=10)
        #dt_model.plot_tree(max_depth=3)
    
    # 5. FINALIZACIÓN
    print(f"\n✅ ANÁLISIS COMPLETO FINALIZADO")
    print("=" * 70)
    print("📁 Archivos generados:")
    print("   Clasificación de Fallos:")
    print("   • feature_importance_clasificacion.png")
    print("   • confusion_matrix.png") 
    print("   • decision_tree_clasificacion.png")
    
    if target_column in df.columns:
        print("   Regresión de Eficiencia:")
        print("   • feature_importance.png")
        print("   • decision_tree.png")
    
    print(f"\n🎯 Ambos modelos completados exitosamente!")
    
    if model_clasificacion is not None and metrics_clasificacion is not None:
        accuracy = metrics_clasificacion['accuracy']
        print(f"📊 Clasificación - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if target_column in df.columns and 'results' in locals():
        r2_score = results['metrics']['r2']
        rmse = results['metrics']['rmse']
        print(f"📈 Regresión - R²: {r2_score:.4f}, RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()