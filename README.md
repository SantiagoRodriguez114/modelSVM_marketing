# modelSVM_marketing

Aplicación de una Máquina de Vectores de Soporte (SVM) para la clasificación de clientes bancarios con el objetivo de predecir si un cliente suscribirá un depósito a plazo (target `y`), basado en información demográfica, de contacto y comportamiento del cliente.

## Descripción del proyecto

Este proyecto desarrolla un flujo completo de análisis y modelado supervisado sobre el dataset de marketing bancario `bank-full.csv`, con foco en la tarea de retención de clientes. La finalidad es identificar clientes con mayor probabilidad de responder positivamente a una campaña de marketing bancario.

Se trabaja con un enfoque realista de negocio y de ciencia de datos:

- Se elimina la variable `duration` por riesgo de fuga de información (data leakage).
- Se trata `pdays = -1` como valor centinela, no como magnitud real.
- Las categorías `"unknown"` se mantienen como categoría válida y no como valor faltante.
- Se usa un split estratificado para preservar la proporción de la clase minoritaria.
- Se incorpora ponderación de clases (`class_weight='balanced'`) para manejar el desbalance del dataset.
- La evaluación se centra en métricas más útiles que accuracy, como precision, recall, F1-score y matriz de confusión.

## Objetivo

Construir y evaluar un modelo SVM que permita:

- predecir si un cliente suscribirá un depósito a plazo,
- equilibrar el rendimiento entre clases desbalanceadas,
- identificar la mejor configuración de hiperparámetros,
- entender los trade-offs entre precisión y sensibilidad según el problema de negocio.

## Contexto del problema

La base de datos corresponde a una campaña de marketing bancario. El objetivo es predecir la variable objetivo:

- `yes`: el cliente sí suscribió un depósito a plazo.
- `no`: el cliente no lo suscribió.

Dado que el problema es de clasificación con clases desbalanceadas (~88%/12%), el uso exclusivo de accuracy es insuficiente. Por eso, la interpretación del modelo se realiza con foco en:

- precisión para la clase positiva,
- recall o sensibilidad para detectar clientes reales que sí responden,
- F1-score para balancear precision y recall,
- análisis de falsos positivos y falsos negativos.

## Datos

El dataset utilizado es:

- `bank-full.csv`

A partir de la información disponible, se realiza:

- codificación categórica mediante `pd.get_dummies`,
- escalado de variables con `StandardScaler`,
- división en entrenamiento y prueba usando `train_test_split` con estratificación,
- uso de una submuestra de ajuste para realizar la exploración de hiperparámetros sin contaminar el conjunto final de prueba.

## Estructura del repositorio

```text
modelSVM_marketing/
├── README.md
├── SVM_marketing.ipynb
└── bank-full.csv        # dataset requerido para ejecutar el notebook
```

## Tecnologías y librerías

El proyecto se desarrolla con Python y las siguientes librerías:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook

## Método de trabajo

### 1. Carga y preparación de datos

Se realiza la carga del dataset con separador `;`:

```python
df = pd.read_csv('bank-full.csv', sep=';')
```

Luego se elimina la variable `duration`, ya que es un dato post-contacto y puede provocar fuga de información.

### 2. Definición de la variable objetivo

Se convierte la variable `y` en una etiqueta binaria:

```python
y = (df['y'] == 'yes').astype(int)
```

### 3. Codificación one-hot

Las variables categóricas se transforman para que el modelo pueda trabajarlos:

```python
X_raw = df.drop(columns=['y'])
X = pd.get_dummies(X_raw, drop_first=True)
```

### 4. Split estratificado

Se separa el dataset en entrenamiento y prueba preservando la proporción de la clase minoritaria:

```python
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### 5. Estandarización

Como SVM es sensible a la escala de las variables, se aplica `StandardScaler`:

```python
scaler = StandardScaler()
Xtr_scaled = scaler.fit_transform(Xtr)
Xte_scaled = scaler.transform(Xte)
```

### 6. Modelo SVM

Se usa `SVC` (Support Vector Classifier) con kernel RBF, ajustando hiperparámetros como:

- `C`: equilibrio entre margen y error de clasificación.
- `gamma`: influencia de cada punto en la frontera de decisión.
- `class_weight='balanced'`: compensate the imbalance in the target class.

```python
base_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    random_state=42
)
```

## Hiperparámetros explorados

El proyecto realiza un estudio manual de varios valores para entender el comportamiento del modelo:

### Parámetro `C`

Se prueba una grilla de valores:

```python
C_values = [0.01, 0.1, 1, 10, 100]
```

Se observa que:

- valores bajos de `C` suelen producir mayor regularización,
- valores altos aumentan la rigidez del modelo y pueden provocar sobreajuste,
- `C = 1` apareció como valor de mejor equilibrio según F1-score en la práctica realizada.

### Parámetro `gamma`

Se evalúa la sensibilidad del modelo frente a distintos valores de gamma:

```python
gamma_values = [0.001, 0.01, 0.1, 1, 'scale']
```

Se concluye que valores muy altos de gamma pueden concentrar la frontera y reducir la capacidad del modelo para detectar la clase positiva, mientras que valores intermedios suelen ofrecer mejor equilibrio.

## Métricas evaluadas

El notebook calcula y compara las siguientes métricas:

- accuracy
- precision
- recall
- f1-score
- confusion matrix
- classification report
- curvas ROC y PR (si se usan en extensiones del notebook)

### Interpretación de resultados

Dado el desbalance del problema, la métrica más representativa para la optimización es F1-score en la clase positiva, ya que reúne precision y recall.

El análisis del proyecto destaca que:

- la exactitud global puede ser alta aunque la detección de clientes que sí responden sea insuficiente,
- una mala elección de hiperparámetros puede aumentar los falsos positivos,
- el objetivo de negocio no es solo “acertar en general”, sino también detectar clientes valiosos para la campaña.

## Resultados obtenidos

El notebook reporta métricas de ejemplo para el modelo base SVM con kernel RBF:

- Accuracy: ~0.823
- Precision: ~0.343
- Recall: ~0.558
- F1-score: ~0.425

Esto indica un modelo funcional, pero con margen importante de mejora si se requiere mayor sensibilidad para detectar clientes potencialmente interesados.

## Recomendaciones de negocio

Dado que el problema es de retención y campañas de marketing, una estrategia útil es:

- priorizar un recall razonable para capturar más clientes positivos,
- revisar el costo de falso positivo vs falso negativo,
- usar el modelo como apoyo de decisión, no como sustituto completo de criterio humano,
- considerar ajustes de umbral de decisión según objetivos de campaña.

## Cómo ejecutar el proyecto

### Opción 1: Jupyter Notebook local

1. Clonar el repositorio:

```bash
git clone https://github.com/SantiagoRodriguez114/modelSVM_marketing.git
cd modelSVM_marketing
```

2. Crear un entorno virtual (opcional pero recomendado):

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

3. Instalar dependencias:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

4. Abrir el notebook:

```bash
jupyter notebook
```

5. Ejecuta `SVM_marketing.ipynb`.

### Opción 2: Google Colab

Puedes subir el notebook a Google Colab y cargar el archivo `bank-full.csv` en el entorno del notebook si no está disponible.

## Requisitos

- Python 3.9+
- Jupyter Notebook o JupyterLab
- Se recomienda entorno virtual para reproducibilidad
- Dataset `bank-full.csv`

## Limitaciones del proyecto

- El dataset no se incluye en este repositorio en esta vista de trabajo, por lo que debe agregarse manualmente para ejecutar el notebook completamene.
- El modelo se ajusta sobre una submuestra para velocidad y exploración, pero el modelo final se valida sobre el conjunto completo de prueba.
- El enfoque principal está en clasificación con desbalance de clases; se recomienda estudiar métricas de costo y umbrales para escenarios de negocio reales.

## Mejoras propuestas

- explorar otros kernels (`linear`, `poly`, `sigmoid`),
- usar `GridSearchCV` para automatizar la búsqueda de hiperparámetros,
- evaluar curvas ROC y PR con umbrales variables,
- comparar con modelos alternativos (Random Forest, XGBoost, Logistic Regression),
- construir un pipeline reproducible con `Pipeline` de scikit-learn,
- documentar resultados en tabla final y análisis de negocio.

## Contribución

Si deseas contribuir al proyecto:

1. Haz un fork del repositorio.
2. Crea una rama para tu mejora.
3. Realiza tus cambios y valida el notebook.
4. Envía un pull request con una descripción clara.

## Autor

Santiago Rodríguez

## Licencia

Este proyecto se distribuye con fines académicos y de aprendizaje. Si se va a reutilizar en un entorno profesional o educativo, es recomendable revisar la licencia del conjunto de datos original y ajustar el uso según el contexto.

## Resumen ejecutivo

Este repositorio presenta una implementación práctica de SVM para clasificación binaria en marketing bancario. El proyecto combina preparación de datos, manejo de clases desbalanceadas, ajuste de hiperparámetros, evaluación con métricas relevantes y análisis de negocio. El objetivo principal es mostrar cómo una SVM puede ser aplicada a un caso real, resaltando la importancia de no centrarse solo en accuracy cuando la clase positiva es minoritaria y estratégicamente relevante.

---

Si quieres, también puedo dejarte una versión aún más profesional del README con:

- badges de Python, scikit-learn y Jupyter,
- secciones de instalación más detalladas,
- tabla de métricas resumidas,
- y un estilo listo para GitHub con formato premium.
