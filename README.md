# 🔬 Análisis de Datos de Talento - Challenge Ingelean

Un sistema completo de análisis de datos para un dataset de manufactura, desarrollado como parte del challenge técnico de Ingelean. El proyecto incluye preprocesamiento de datos, análisis de correlaciones y generación de visualizaciones.

## 📊 Descripción del Proyecto

Este proyecto analiza un dataset de manufactura con **6,000 registros** y **18 variables** que incluyen:
- Variables operacionales (temperatura, vibración, humedad, tiempo de ciclo)
- Métricas de producción (cantidad producida, unidades defectuosas, eficiencia)
- Información de fallos y paradas
- Datos temporales y de identificación

### 🎯 Objetivos Principales
- **Análisis exploratorio** de datos de manufactura
- **Preprocesamiento** y limpieza de datos
- **Análisis de correlaciones** entre variables operacionales
- **Generación de insights** para optimización de procesos

## 🚀 Instalación y Configuración

### Requisitos Previos
- Python 3.7+
- pip (gestor de paquetes de Python)

### Instalación
```bash
# Clonar el repositorio
git clone https://github.com/Fabian0594/ingelean_challenge.git
cd ingelean_challenge

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Configuración
Crear archivo `.env` en la raíz del proyecto:
```env
rute=Dataset_Talento.csv
```

## 📁 Estructura del Proyecto

```
ingelean_challenge/
├── main.py                        # Script principal de análisis
├── process_data.py                 # Clase para procesamiento de datos
├── Dataset_Talento.csv            # Dataset principal (6K registros)
├── requirements.txt               # Dependencias del proyecto
├── .env                          # Variables de configuración
├── README.md                     # Documentación del proyecto
├── 📊 Archivos Generados/
│   ├── matriz_correlaciones.csv      # Matriz completa de correlaciones
│   ├── top_correlaciones.csv         # Top correlaciones ordenadas
│   ├── heatmap_correlaciones.png     # Mapa de calor visual
│   └── reporte_perfilado.html        # Reporte de perfilado (opcional)
└── venv/                         # Entorno virtual
```

## ⚙️ Funcionalidades Implementadas

### 🔧 Preprocesamiento de Datos
- **Limpieza de nombres de columnas**: Normalización y eliminación de caracteres especiales
- **Creación de variables binarias**: Transformación de fallos en variables categóricas
- **Imputación de valores faltantes**: Estrategias de media y mediana para variables numéricas

### 📈 Análisis de Correlaciones
- **Matriz de correlación de Pearson** completa (11×11 variables numéricas)
- **Identificación de correlaciones significativas** ordenadas por valor absoluto
- **Análisis estadístico** con métricas de resumen
- **Visualización mediante mapas de calor**

### 📋 Generación de Reportes
- **Archivos CSV** con matrices de correlación
- **Visualizaciones PNG** con mapas de calor
- **Reportes HTML** de perfilado de datos (opcional)

## 📊 Resultados Principales

### 🔍 Variables Analizadas
El análisis se realizó sobre **11 variables numéricas**:
- `temperatura`, `vibracion`, `humedad`, `tiempo_ciclo`
- `cantidad_producida`, `unidades_defectuosas`, `eficiencia_porcentual`
- `consumo_energia`, `paradas_programadas`, `paradas_imprevistas`
- `fallos_binarios` (variable creada)

### 📈 Top 5 Correlaciones Más Altas

| Ranking | Variable 1 | Variable 2 | Correlación | Interpretación |
|---------|------------|------------|-------------|----------------|
| 1 | `fallos_binarios` | `tiempo_ciclo` | **0.04** | Los fallos se asocian con ciclos más largos |
| 2 | `unidades_defectuosas` | `temperatura` | **-0.04** | Mayor temperatura → menos defectos |
| 3 | `paradas_programadas` | `tiempo_ciclo` | **-0.04** | Más paradas programadas → ciclos más cortos |
| 4 | `unidades_defectuosas` | `tiempo_ciclo` | **-0.03** | Ciclos largos → menos defectos |
| 5 | `fallos_binarios` | `unidades_defectuosas` | **0.02** | Fallos correlacionan con defectos |

### 🎯 Insights Clave
- **Correlaciones débiles**: Todas las correlaciones son menores a 0.05, indicando **independencia** entre variables
- **Tiempo de ciclo** es la variable más correlacionada con otras variables operacionales
- **Temperatura** muestra relación inversa con defectos de producción
- **Los datos presentan buena calidad** para modelos de machine learning (variables independientes)

### 📊 Estadísticas de Correlación
- **Total de correlaciones únicas**: 55 pares
- **Correlación máxima**: 0.0418 (`fallos_binarios` ↔ `tiempo_ciclo`)
- **Correlación promedio**: 0.0120
- **Correlaciones > 0.1**: 0 (ninguna correlación fuerte)

## 🖥️ Uso del Sistema

### Ejecución Básica
```bash
python main.py
```

### Funciones Principales

#### Análisis Completo de Correlaciones
```python
from main import crear_tabla_correlaciones

# Análisis completo con archivos y visualización
matriz = crear_tabla_correlaciones(df, 
                                  guardar_archivo=True, 
                                  mostrar_heatmap=True)
```

#### Preprocesamiento de Datos
```python
from main import crear_fallos_binarios, imputacion_simple

# Crear variables binarias de fallos
df_con_fallos = crear_fallos_binarios(df, 'tipo_fallo')

# Imputar valores faltantes
df_limpio = imputacion_simple(df_con_fallos, estrategia='media')
```

## 📁 Archivos de Salida

### 📋 `matriz_correlaciones.csv`
Matriz completa 11×11 con todas las correlaciones de Pearson entre variables numéricas.

### 📊 `top_correlaciones.csv`
```csv
Variable_1,Variable_2,Correlacion
fallos_binarios,tiempo_ciclo,0.04
unidades_defectuosas,temperatura,-0.04
paradas_programadas,tiempo_ciclo,-0.04
...
```

### 🎨 `heatmap_correlaciones.png`
Mapa de calor visual de la matriz de correlaciones con:
- Escala de colores **coolwarm** (-1 a +1)
- Anotaciones numéricas en cada celda
- Resolución alta (300 DPI) para análisis detallado

## 🛠️ Tecnologías Utilizadas

- **Python 3.7+**: Lenguaje principal
- **pandas**: Manipulación y análisis de datos
- **numpy**: Cálculos numéricos
- **matplotlib**: Visualizaciones básicas
- **seaborn**: Visualizaciones estadísticas avanzadas
- **scikit-learn**: Herramientas de machine learning
- **python-dotenv**: Gestión de variables de entorno

## 📈 Métricas del Proyecto

- **Líneas de código**: ~340 líneas
- **Cobertura de datos**: 6,000 registros, 18 variables
- **Variables numéricas analizadas**: 11
- **Correlaciones calculadas**: 55 pares únicos
- **Archivos generados**: 3 tipos de salida
- **Tiempo de ejecución**: < 10 segundos

## 🔍 Análisis de Calidad de Datos

### Valores Faltantes (Antes del Preprocesamiento)
- `temperatura`: 180 (3%)
- `vibracion`: 180 (3%)
- `humedad`: 180 (3%)
- `tiempo_ciclo`: 180 (3%)
- `eficiencia_porcentual`: 180 (3%)
- `consumo_energia`: 180 (3%)
- `tipo_fallo`: 5,401 (90%) - Por diseño
- `observaciones`: 4,226 (70%) - Datos opcionales

### Distribución de Fallos
- **Sin fallo**: 5,401 registros (90.0%)
- **Con fallo**: 599 registros (10.0%)

## 🚀 Próximos Pasos

1. **Análisis Predictivo**: Implementar modelos de machine learning para predicción de fallos
2. **Análisis Temporal**: Estudiar patrones temporales en los datos
3. **Segmentación**: Análisis por turnos, operadores y máquinas
4. **Optimización**: Identificar factores clave para mejora de eficiencia

## 👨‍💻 Autor

**Fabián Rodriguez**
- GitHub: [@Fabian0594](https://github.com/Fabian0594)
- Proyecto: Challenge Técnico Ingelean

## 📄 Licencia

Este proyecto fue desarrollado como parte de un challenge técnico para Ingelean.

---

⭐ **¿Te resultó útil este análisis?** ¡Dale una estrella al repositorio!