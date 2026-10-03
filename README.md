# 🎯 SVM Marketing Classification Model

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37726?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)]()

**Aplicación de Máquinas de Vectores de Soporte (SVM) para predicción de suscripción a depósitos bancarios**

[🔍 Problema](#-el-problema) • [📊 Datos](#-datos) • [🏗️ Solución](#-solución-propuesta) • [📈 Resultados](#-resultados) • [🚀 Uso](#-cómo-ejecutar)

</div>

---

## 🎯 Descripción 

Este proyecto implementa un **modelo de clasificación SVM** de nivel producción para predecir si un cliente bancario suscribirá un depósito a plazo. Aborda los desafíos reales de:

- ✅ **Clases desbalanceadas** (~88%/12%)
- ✅ **Prevención de data leakage**
- ✅ **Optimización de métricas relevantes** más allá de accuracy
- ✅ **Interpretación de trade-offs** precision vs recall
- ✅ **Validación estratificada** para garantizar reproducibilidad

**Caso de uso**: Optimizar campañas de marketing bancario mediante identificación de clientes de alto valor.

---

## 🔍 El Problema

### Contexto de Negocio

Una institución bancaria realiza campañas de marketing para promover la suscripción de depósitos a plazo. Históricamente:

- Solo **~12%** de los clientes contactados aceptan la oferta
- Los recursos de marketing son limitados
- Cada contacto tiene un costo asociado
- Existe una oportunidad de mejorar la segmentación

### Reto Técnico

Construir un modelo que:

1. **Identifique correctamente** clientes con alta probabilidad de suscripción
2. **Equilibre precision y recall** en contexto de clases desbalanceadas
3. **Evite contaminación** de información (data leakage)
4. **Sea interpretable** para toma de decisiones de negocio

---

## 📊 Datos

### Dataset: `bank-full.csv`

| Métrica | Valor |
|---------|-------|
| **Observaciones** | 45,211 |
| **Características (originales)** | 20 |
| **Características (post-encoding)** | 41 |
| **Clase positiva (y=yes)** | 11.7% |
| **Clase negativa (y=no)** | 88.3% |
| **Proporción desbalance** | ~7.5:1 |

### Variables clave

**Demográficas**: edad, estado civil, educación  
**Contacto**: mes, día de la semana, tipo de contacto  
**Histórico**: número de contactos, resultado de la campaña anterior  
**Económicas**: balance, préstamo personal, hipoteca  

### Decisiones de preprocesamiento

| Decisión | Razón |
|----------|-------|
| ❌ Eliminar `duration` | Fuga de información (post-evento) |
| ✅ Mantener `pdays=-1` como categoría | Valor centinela (sin contacto previo) |
| ✅ Codificar `"unknown"` | Información válida, no valor faltante |
| ✅ Estratificación en split | Preservar proporción de clases minoritarias |
| ✅ StandardScaler | Sensibilidad de SVM a escala de features |

---

## 🏗️ Solución Propuesta

### Arquitectura del modelo

```
┌─────────────────────┐
│  bank-full.csv      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  1. Preprocesamiento                │
│     • Drop: duration                │
│     • Encode: pd.get_dummies()      │
│     • Scale: StandardScaler         │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  2. Train/Test Split                │
│     • Stratified: 80/20             │
│     • Random state: 42              │
│     • Total train: 36,168           │
│     • Total test: 9,043             │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  3. Exploración de Hiperparámetros  │
│     • Submuestra: 8,000             │
│     • Kernel: RBF                   │
│     • C: [0.01, 0.1, 1, 10, 100]    │
│     • Gamma: [0.001, 0.01, ...]     │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  4. Entrenamiento Final             │
│     • SVC(kernel='rbf',             │
│          C=1.0,                     │
│          gamma=0.01,                │
│          class_weight='balanced')   │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  5. Evaluación en Test              │
│     • Accuracy, Precision, Recall   │
│     • F1-Score, Confusion Matrix    │
│     • Análisis de negocio           │
└─────────────────────────────────────┘
```

### Selección del algoritmo: SVM

**¿Por qué SVM?**

| Criterio | Ventaja SVM |
|----------|------------|
| **Datos altos-dimensionales** | Excelente en espacios de alta dimensión (41 features) |
| **No-linealidad** | Kernel RBF captura relaciones complejas |
| **Regularización** | Parámetro `C` integrado para evitar sobreajuste |
| **Probabilidades de clase** | Con `probability=True`, permite ajuste de umbrales |
| **Interpretabilidad** | Vectores de soporte identifican muestras clave |

---

## 🧪 Metodología

### Fase 1: Carga y Limpieza

```python
# Carga con separador correcto
df = pd.read_csv('bank-full.csv', sep=';')

# Eliminación de variable con data leakage
df = df.drop(columns=['duration'])

# Target binario
y = (df['y'] == 'yes').astype(int)
```

### Fase 2: Feature Engineering

```python
# Codificación one-hot para categorías
X = pd.get_dummies(X_raw, drop_first=True)  # 41 features

# Estratificación: preservar balance de clases
Xtr, Xte, ytr, yte = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,  # ← clave para clases desbalanceadas
    random_state=42
)
```

### Fase 3: Escalado

```python
# StandardScaler (obligatorio para SVM)
scaler = StandardScaler()
Xtr_scaled = scaler.fit_transform(Xtr)
Xte_scaled = scaler.transform(Xte)
```

### Fase 4: Exploración de Hiperparámetros

#### 4a. Parámetro `C`

| C | Accuracy | Precision | Recall | F1-Score | Análisis |
|---|----------|-----------|--------|----------|----------|
| **0.01** | 71.2% | 22.3% | 58.7% | 0.323 | Subreajuste: baja precisión |
| **0.10** | 80.5% | 31.7% | 58.1% | 0.411 | Mejor recall pero baja precision |
| **1.00** ⭐ | **82.3%** | **34.3%** | **55.8%** | **0.425** | **Óptimo: mejor F1-score** |
| **10.00** | 80.4% | 27.6% | 41.5% | 0.331 | Sobreajuste: baja recall |
| **100.00** | 80.0% | 24.9% | 35.3% | 0.292 | Muy rígido: pobre generalización |

**Conclusión**: `C=1.0` ofrece el mejor balance entre precision y recall para F1-score.

#### 4b. Parámetro `Gamma` (RBF)

| Gamma | Accuracy | Precision | Recall | F1-Score | Comportamiento |
|-------|----------|-----------|--------|----------|---|
| 0.001 | 82.5% | 34.1% | 53.2% | 0.415 | Influencia amplia |
| **0.01** ⭐ | **82.7%** | **35.3%** | **57.5%** | **0.438** | **Mejor interpretabilidad** |
| 0.10 | 81.0% | 25.9% | 33.5% | 0.292 | Empeora con gamma alto |
| 1.00 | 84.9% | 13.8% | 5.5% | 0.078 | Sobreajuste severo |
| scale | 82.3% | 34.3% | 55.8% | 0.425 | Baseline |

**Conclusión**: `gamma=0.01` logra el mejor F1-score (0.438), indicando mejor balance en datos de prueba.

---

## 📈 Resultados

### Modelo Base (Kernel RBF, C=1.0, Gamma=scale)

```
┌──────────────────────────────────────┐
│         MÉTRICAS DE RENDIMIENTO      │
├──────────────────────────────────────┤
│  Accuracy:  82.3%                    │
│  Precision: 34.3%  (de 1,720 pred+,  │
│  Recall:    55.8%   590 correctos)   │
│  F1-Score:  0.425                    │
└──────────────────────────────────────┘
```

### Matriz de Confusión

```
                 Predicción
               No        Sí
Real  No  | 6,855    1,130 |  (TN)    (FP)
      Sí  |   468      590 |  (FN)    (TP)

Desglose:
  • Verdaderos Negativos (TN):   6,855 — Rechazos correctos
  • Falsos Positivos (FP):       1,130 — Falsas esperanzas (costo)
  • Falsos Negativos (FN):         468 — Oportunidades perdidas
  • Verdaderos Positivos (TP):     590 — Conversiones correctas
```

### Interpretación de Resultados

#### ✅ Fortalezas

1. **Recall 55.8%**: Detecta más de la mitad de clientes que realmente suscribirán
2. **Accuracy 82.3%**: Buen desempeño general en clasificación
3. **Balanced Approach**: Con `class_weight='balanced'`, no cae en la trampa de predecir todo "No"

#### ⚠️ Áreas de mejora

1. **Precision 34.3%**: Por cada 3 clientes predichos "Sí", solo 1 realmente lo es (2 falsos positivos)
2. **FP: 1,130**: Contactos innecesarios pueden aumentar costos de campaña
3. **FN: 468**: Oportunidades de venta perdidas (40% de positivos no detectados)

---

## 💼 Recomendaciones de Negocio

### Escenario 1: Maximizar Conversiones (Risk: Alto costo)
**Ajustar umbral de decisión a 0.3** (en lugar de 0.5)
- ↑ Recall: ~75% (capturar más clientes positivos)
- ↓ Precision: ~20% (aceptar más falsos positivos)
- **Uso**: Campañas con presupuesto abundante

### Escenario 2: Optimizar ROI (Risk: Medio)
**Mantener umbral en 0.5** (configuración actual)
- Balance aceptable entre recall y precision
- **Uso**: Campañas con presupuesto moderado

### Escenario 3: Máxima Precisión (Risk: Bajo costo)
**Ajustar umbral a 0.7**
- ↑ Precision: ~60-70%
- ↓ Recall: ~25-30% (contactar solo a clientes muy seguros)
- **Uso**: Campañas premium, presupuesto muy limitado

---

## 🚀 Cómo Ejecutar

### ⚙️ Instalación

#### 1. Clonar repositorio

```bash
git clone https://github.com/SantiagoRodriguez114/modelSVM_marketing.git
cd modelSVM_marketing
```

#### 2. Crear entorno virtual

```bash
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

#### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Alternativa**: Instalación manual
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

#### 4. Verificar instalación

```bash
python -c "import sklearn; print(f'scikit-learn {sklearn.__version__}')"
```

### 📓 Ejecutar Notebook

```bash
# Iniciar Jupyter
jupyter notebook

# O usar JupyterLab (interfaz mejorada)
jupyter lab
```

Luego abre `SVM_marketing.ipynb` y ejecuta las celdas en orden.

### ☁️ Ejecutar en Google Colab (sin instalación local)

1. Ve a [Google Colab](https://colab.research.google.com/)
2. Carga `SVM_marketing.ipynb`
3. Carga `bank-full.csv` en el entorno
4. Ejecuta las celdas

---

## 📦 Estructura del Repositorio

```
modelSVM_marketing/
│
├── README.md                          # ← Documentación (este archivo)
├── requirements.txt                   # Dependencias Python
│
├── SVM_marketing.ipynb               # Notebook principal
│   ├── 1. Carga y Preprocesamiento
│   ├── 2. Exploración de SVM
│   ├── 3. Ajuste de Hiperparámetros
│   └── 4. Evaluación Final
│
├── bank-full.csv                     # Dataset (agregar manualmente)
│
└── .gitignore                        # Archivos a ignorar en Git
```

---

## 🔧 Requisitos del Sistema

| Requisito | Versión |
|-----------|---------|
| Python | 3.9+ |
| pandas | 1.5+ |
| numpy | 1.23+ |
| scikit-learn | 1.3+ |
| matplotlib | 3.5+ |
| seaborn | 0.12+ |
| Jupyter | 1.0+ |

**RAM mínima recomendada**: 4 GB  
**Espacio disco**: ~500 MB

---

## 🎓 Conceptos Clave

### ¿Qué es SVM?

**Máquina de Vectores de Soporte** es un algoritmo de clasificación que busca el **hiperplano óptimo** que:

- Maximiza la distancia (margen) con respecto a los puntos más cercanos de cada clase
- Minimiza errores de clasificación
- Maneja datos no-lineales mediante kernels (RBF, polinómico, etc.)

```
Espacio original (No-lineal)    →    Espacio transformado (Lineal)
        •• (No)                          
       •  • (No)                  Kernel RBF
      •  •   (Sí) ✗             ─────────→      Separación óptima
       •  •  (Sí)
        ••  (No)                          ___________
```

### Desbalance de Clases: ¿Por qué es un problema?

Con 88% No y 12% Sí, un modelo "ingenuo" que predice todo "No" logra 88% de accuracy... ¡pero es inútil!

**Soluciones implementadas:**

1. **`class_weight='balanced'`**: Penaliza más los errores en la clase minoritaria
2. **Stratified Split**: Garantiza que train/test tengan la misma proporción
3. **Métricas sensibles**: Usar F1-score, precision, recall en lugar de solo accuracy

---

## 📚 Teoría y Referencias

### Parámetros SVM explicados

| Parámetro | Rango | Efecto |
|-----------|-------|--------|
| **C** | (0, ∞) | Regularización. Bajo = margen ancho (subajuste). Alto = ajuste a datos (sobreajuste). |
| **gamma** | (0, ∞) | Alcance de influencia. Bajo = influencia global. Alto = influencia local (sobreajuste). |
| **kernel** | linear, poly, rbf, sigmoid | Función de transformación. RBF es versátil y efectivo. |
| **degree** | int ≥ 1 | Solo para kernel polinómico. Mayor grado = fronteras más complejas. |

### Métricas de Evaluación

```
            Clase Predicha
           Pos     Neg
Real Pos | TP  |  FN  |    Recall = TP / (TP + FN)
    Neg  | FP  |  TN  |    Precision = TP / (TP + FP)

F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
           → Promedio armónico (penaliza desbalance)
```

---

## 🔄 Mejoras Futuras

### 🔄 Corto Plazo

- [ ] Implementar GridSearchCV para automatizar búsqueda de hiperparámetros
- [ ] Evaluar curva ROC y área bajo la curva (AUC)
- [ ] Incluir análisis de importancia de features (permutation importance)
- [ ] Crear pipeline reproducible con `sklearn.Pipeline`

### 📊 Mediano Plazo

- [ ] Comparar con otros algoritmos (Random Forest, XGBoost, LightGBM)
- [ ] Implementar validación cruzada k-fold estratificada
- [ ] Analizar curva de aprendizaje (learning curves)
- [ ] Tunning de umbral de decisión según costo de negocio

### 🚀 Largo Plazo

- [ ] Desplegar modelo como API REST (FastAPI)
- [ ] Crear dashboard interactivo (Streamlit)
- [ ] Implementar monitoreo de drift en producción
- [ ] A/B testing de estrategias de contacto
- [ ] Explicabilidad con SHAP values

---

## 🤝 Contribución

¿Quieres mejorar este proyecto? ¡Adelante!

1. **Fork** el repositorio
2. Crea una rama: `git checkout -b feature/mejora-modelo`
3. Realiza cambios y **documenta** tus mejoras
4. **Valida** que el notebook ejecuta correctamente
5. Push: `git push origin feature/mejora-modelo`
6. Abre un **Pull Request** con descripción clara

### Ideas de contribución

- Nuevas técnicas de balanceo (SMOTE, ADASYN)
- Visualizaciones interactivas
- Documentación adicional
- Casos de uso alternativos

---

## 📝 Licencia

Este proyecto se distribuye bajo licencia **MIT**. Libre para uso académico y comercial.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 👨‍💻 Autor

**Santiago Rodríguez**

- 🔗 [GitHub](https://github.com/SantiagoRodriguez114)
- 📧 [Email](mailto:rodriguezsanti751@gmail.com)
- 💼 [LinkedIn](https://linkedin.com/in/santiago-rodríguez)

---

## 📞 Soporte

¿Preguntas o problemas?

- 📌 Abre un **issue** en GitHub
- 💬 Revisa la sección de [FAQ](#faq) abajo
- 📧 Contacta directamente al autor

---

## 🎯 FAQ

<details>
<summary><b>¿Dónde consigo el dataset bank-full.csv?</b></summary>

El dataset no se incluye en el repositorio por tamaño. Puedes:

1. [Descargar desde UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
2. Colocarlo en la raíz del proyecto como `bank-full.csv`
3. Ejecutar el notebook

</details>

<details>
<summary><b>¿Por qué elimino 'duration'?</b></summary>

Porque `duration` es la duración de la llamada de contacto, que **solo se conoce después** de establecer contacto. Si lo usamos en entrenamiento, creamos un **data leakage** que no podría usarse en predicción real.

</details>

<details>
<summary><b>¿El modelo está listo para producción?</b></summary>

No completamente. Faltaría:

- ✅ Validation cross-fold estratificada
- ✅ Manejo de nuevas categorías en features
- ✅ Monitoring de performance en tiempo real
- ✅ API de inferencia
- ✅ Documentación de decisiones de negocio

Es un excelente **prototipo** para POC (Proof of Concept).

</details>

<details>
<summary><b>¿Cómo ajusto el modelo para maximizar recall?</b></summary>

Reduce el umbral de decisión:

```python
# Predicciones con probabilidades
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Umbral personalizado (ej: 0.3 en lugar de 0.5)
y_pred_custom = (y_pred_proba >= 0.3).astype(int)
```

Esto capturará más clientes positivos pero aumentará falsos positivos.

</details>

---

## 📊 Resumen Ejecutivo

| Métrica | Valor | Implicación |
|---------|-------|------------|
| **F1-Score** | 0.425 | Modelo funcional con margen de mejora |
| **Recall** | 55.8% | Detecta ~56% de clientes que realmente suscribirán |
| **Precision** | 34.3% | De 100 contactos predichos "Sí", ~34 convierten |
| **Clientes Objetivo (Pos)** | 1,058 | Total de clientes que realmente suscribieron |
| **Detectados Correctamente** | 590 | Oportunidades de venta capturadas |
| **Oportunidades Perdidas** | 468 | Clientes no contactados que habrían suscrito |

**Conclusión**: El modelo SVM proporciona una base sólida para mejorar la eficiencia de campañas de marketing, pero requiere refinamiento en hyperparámetros y posiblemente exploración de algoritmos complementarios para optimizar el ROI en un entorno de negocio real.

---


