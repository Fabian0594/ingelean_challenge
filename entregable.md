# HACKATHÓN TALENTO TECH – PLANTILLA OFICIAL DEL RETO

---

## 🏆 **NOMBRE DEL EQUIPO**
*Nombre creativo y representativo*

**Grupo 3 Pa 4.**

El logo no solo representa a los 3 miembros del grupo como individuos, sino evolucionando hacia un nuevo nivel de cambio y crecimiento.

---

## 🚀 **NOMBRE DEL PROYECTO**
*Breve, original, relacionado con análisis y predicción*

**Análisis de variables industriales y su impacto en la eficiencia y producción.**

---

## 📋 **DESCRIPCIÓN GENERAL DEL PROYECTO** *(MÁX. 100 PALABRAS)*
*¿Qué problema empresarial aborda su solución? ¿Qué variable están intentando predecir? ¿Qué tipo de empresa o sector imaginaron?*

La solución aborda el mejoramiento de la eficiencia y producción en una empresa al determinar las falencias en el funcionamiento de máquinas, operarios y turnos. Se intenta predecir los tipos de fallos que presentan una falta de datos considerable. Igualmente, se intenta determinar si una cantidad de datos nulos (180) en varios datos se corresponden entre sí o si son aleatoriamente coincidentes. Por el tipo de variables descritas en el DATASET (temperatura en promedio 70°, vibración, humedad, diferentes productos en una misma máquina), se puede inferir que la empresa está dedicada a la creación de productos plásticos por inyección, probablemente.

---

## 📊 **DATASET UTILIZADO**
*¿De dónde se obtuvo? (proporcionado / simulación / fuente externa) ¿Qué variables contiene? ¿Cuántos registros y columnas tiene? ¿Qué tipo de limpieza o preprocesamiento aplicaron?*

**Origen:** El DATASET fue proporcionado.

### **Variables que contiene:**
- **Timestamp:** Fecha año-mes-día hora:minuto:segundo
- **Turno:** mañana, tarde y noche (3 turnos)
- **operador_id:** OP_x (número)
- **maquina_id:** M_x (número)
- **producto_id:** P_x (número)
- **temperatura:** x.xxxxxxx (14 decimales) (sin unidades)
- **vibración:** x.xxxx (16 decimales) (sin unidades)
- **humedad:** x.xxxx (15 decimales) (sin unidades)
- **tiempo_ciclo:** xxx.xxxx (14 decimales) (sin unidades)
- **fallo_detectado:** Sí o No
- **tipo_fallo:** nan, u otro (Eléctrico, mecánico...)
- **cantidad_producida:** xxx (sin unidades)
- **unidades_defectuosas:** xx (sin unidades)
- **eficiencia_porcentual:** x.xxxx (14 decimales) (sin unidades)
- **consumo_energia:** x.xxxx (15 decimales) (sin unidades)
- **paradas_programadas:** x
- **paradas_imprevistas:** x
- **observaciones:** nan, u otro (Operación normal, revisar calibración...)

### **Dimensiones:**
- **18 columnas** y **6000 registros**

### **Tipo de limpieza:**
- Eliminación de espacios, caracteres especiales y tildes
- Estructuración a través de Profile_Report en Python para análisis e identificación de tipos de datos y faltantes
- Para los datos faltantes (no superaban el 10% de la base y el 3% de faltantes por columna), se realizó imputación de datos a través de la media aritmética

---

## 🤖 **MODELO PREDICTIVO IMPLEMENTADO**
*¿Qué algoritmo utilizaron? (Regresión lineal, Árbol de decisión, etc.) ¿Qué variable predijeron? ¿Cómo fue el rendimiento del modelo? (Errores, precisión, etc.) ¿Qué supuestos hicieron?*

### **Algoritmo utilizado:**
Árbol de decisión

### **Variables predichas:**
1. **Eficiencia porcentual** (regresión)
2. **Fallos binarios** (clasificación)

### **Rendimiento del modelo:**
- **Clasificación de fallos:** Precisión del 89.11%
- **Regresión de eficiencia:** R² = -0.0287

### **Top de importancia** (variables numéricas):
**Para detección de fallos:**
1. Humedad (30.26%)
2. Temperatura (20.79%)
3. Vibración (16.19%)
4. Tiempo de ciclo (10.30%)

**Para predicción de eficiencia:**
1. Tiempo de ciclo (24.40%)
2. Cantidad producida (18.82%)
3. Temperatura (15.39%)

---

## 📈 **VISUALIZACIONES**
*¿Qué tipo de gráficos generaron? ¿Qué muestran las gráficas? ¿Cómo se comparan las predicciones con los datos reales? ¿Qué herramienta usaron? (Matplotlib, Seaborn, Power BI, etc.)*

### **Herramientas utilizadas:**
- **Python:** Matplotlib, Seaborn, pandas profiling
- **ydata-profiling** para generación de reportes automatizados

### **Visualizaciones generadas:**

#### **1. Summary Report (Profile_Report)**
Inicialmente se generó un Summary a través de un Profile_Report de Python. A través del Summary se pudo observar estadísticas pertinentes como:
- Número de variables
- Número de observaciones
- Celdas perdidas
- Gran cantidad de tipos de fallo no especificados
- Gran cantidad de observaciones sin datos

#### **2. Visualizaciones específicas:**
- **Mapa de calor de correlaciones:** Muestra las relaciones entre variables
- **Matriz de confusión:** Evaluación del modelo de clasificación
- **Gráfico de importancia de características:** Variables más relevantes para predicción
- **Árbol de decisión:** Visualización del modelo y reglas de decisión
- **Análisis de distribución de clases:** Desbalance entre fallos y no fallos

---

## 💡 **RECOMENDACIONES ESTRATÉGICAS**
*Redacten al menos 2 recomendaciones empresariales derivadas del análisis. Deben estar justificadas con base en las predicciones y gráficas.*

### **Recomendación 1: Optimización de Parámetros de Proceso**
Para mejorar la eficiencia porcentual se recomienda bajar los tiempos de ciclo a valores ≤133.783, y si adicional a esto se mantiene la humedad ≤80.147, aumenta la eficiencia porcentual sin que los fallos afecten.

### **Recomendación 2: Sistema de Detección Temprana de Fallos**
Implementar un sistema de monitoreo en tiempo real enfocado en las variables más críticas (humedad, temperatura y vibración) que permita detectar patrones anómalos antes de que ocurran fallos, reduciendo el actual 10% de incidencia de fallos y mejorando la detección que actualmente solo identifica correctamente el 1.05% de los fallos reales.

---

## 🌟 **VALOR DIFERENCIAL DE LA SOLUCIÓN** *(MÁX. 80 PALABRAS)*
*¿Qué hace único su enfoque? ¿Cómo podría escalarse o mejorarse esta solución en una siguiente etapa?*

Se presentan diferentes opciones y generan caminos según las metas, las cuales se bifurcan a medida que se obtienen resultados. El enfoque único combina análisis de correlaciones débiles con modelos especializados para diferentes objetivos (clasificación y regresión), proporcionando insights tanto para mantenimiento predictivo como para optimización de eficiencia. La solución es escalable mediante integración IoT, modelos ensemble y sistemas de alertas en tiempo real.

---

## 🔗 **ENLACES DE ENTREGA**

- **Repositorio de código (GitHub):** [https://github.com/Fabian0594/ingelean_challenge](https://github.com/Fabian0594/ingelean_challenge)
- **Notebook / Dashboard interactivo:** [reporte_perfilado.html](./reporte_perfilado.html)
- **Video demo (opcional, máx. 3 minutos):** _[Por definir]_

---

## 🎤 **PITCH FINAL**
*Estructura sugerida para la presentación:*

### **⏱️ Problema y variable a predecir (30s)**
Optimización de eficiencia industrial y detección predictiva de fallos en proceso de manufactura con 10% de incidencia de fallos y correlaciones débiles entre variables.

### **📊 Análisis de datos y modelo implementado (1 min)**
Dataset de 6,000 registros con 18 variables, aplicando árbol de decisión para clasificación de fallos (89.11% accuracy) y regresión de eficiencia. Variables clave: humedad, temperatura y vibración para fallos; tiempo de ciclo y cantidad producida para eficiencia.

### **📈 Visualizaciones clave y resultados (1 min)**
Mapa de correlaciones revela independencia de variables (máx. 0.04), matriz de confusión muestra problema crítico de recall (1.05%), y gráficos de importancia identifican humedad como factor dominante (30.26%).

### **💼 Recomendaciones y posible impacto (30s)**
Reducir tiempo de ciclo ≤133.783 y mantener humedad ≤80.147 para optimizar eficiencia. Implementar monitoreo predictivo puede generar ROI 200-500% reduciendo paradas imprevistas 15-25%.

### **🚀 Cierre creativo o llamado a la acción (30s)**
"De reactivo a predictivo: transformando datos industriales en ventaja competitiva. ¡El futuro de la manufactura inteligente empieza hoy!"

---

## 📋 **ENTREGABLES ESPERADOS**

### ✅ **Completados:**
- [x] **Dataset limpio y documentado** - `Dataset_Talento_Procesado.csv`
- [x] **Notebook o script de análisis** - `main.py`, `decision_tree.py`
- [x] **Mínimo 2 visualizaciones** - 6 visualizaciones generadas
- [x] **Reporte breve con 2 recomendaciones estratégicas** - `conclusiones_y_recomendaciones.md`
- [x] **Repositorio en GitHub organizado** - Estructura completa con documentación
- [ ] **Pitch final en vivo** (opcional: video corto)

### 📁 **Estructura del repositorio:**
```
ingelean_challenge/
├── main.py                              # Análisis de correlaciones
├── decision_tree.py                     # Modelos de árbol de decisión  
├── process_data.py                      # Utilidades de procesamiento
├── Dataset_Talento.csv                  # Dataset original
├── Dataset_Talento_Procesado.csv        # Dataset limpio
├── reporte_perfilado.html              # Reporte automatizado
├── heatmap_correlaciones.png           # Mapa de calor
├── confusion_matrix.png                # Matriz de confusión
├── decision_tree_clasificacion.png     # Árbol de decisión
├── feature_importance_clasificacion.png # Importancia de características
├── requirements.txt                     # Dependencias
├── README.md                           # Documentación
└── entregable.md                       # Plantilla oficial (este archivo)
```

---

*Hackathón Talento Tech - Enero 2025*  
*Grupo 3 Pa 4 - Análisis de Variables Industriales* 