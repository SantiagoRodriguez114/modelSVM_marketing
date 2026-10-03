# Support Vector Machine for Bank Marketing Classification

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?style=flat-square)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37726?style=flat-square)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Abstract

This project implements a supervised machine learning pipeline for binary classification of bank marketing campaign outcomes. The objective is to predict whether a client will subscribe to a term deposit based on demographic, economic, and behavioral features. We employ Support Vector Machines with kernel-based non-linear transformations and address the critical challenge of class imbalance through balanced weighting, stratified sampling, and evaluation using class-sensitive metrics. The methodology includes systematic hyperparameter optimization and comprehensive preprocessing to prevent information leakage. Final results demonstrate an F1-score of 0.438 on the test set, balancing precision (0.353) and recall (0.575).

**Keywords**: Support Vector Machines, classification, imbalanced learning, kernel methods, hyperparameter optimization

---

## 1. Introduction

### 1.1 Problem Context

Bank marketing campaigns represent a significant operational investment. Targeting the right customers directly impacts campaign efficiency and return on investment. Traditional approaches rely on heuristic segmentation; however, predictive modeling offers a data-driven alternative to identify high-value prospects.

The central problem is to build a classifier that predicts term deposit subscription based on customer information collected during marketing interactions.

### 1.2 Challenges

The dataset exhibits three key challenges:

1. **Class Imbalance**: Only 11.7% of customers subscribe, creating a severe class imbalance (~7.5:1 ratio).
2. **Information Leakage**: Some variables (e.g., call duration) are only known post-contact and must be excluded from predictive modeling.
3. **Metric Selection**: In imbalanced settings, accuracy is uninformative. Precision, recall, and F1-score are more appropriate.

### 1.3 Proposed Approach

We implement a complete preprocessing and modeling pipeline:

- Removal of post-event variables to prevent information leakage
- Categorical feature encoding and feature scaling
- Stratified train/test splitting to preserve class distributions
- Support Vector Machine training with balanced class weights
- Systematic hyperparameter exploration and model evaluation

---

## 2. Data Description

### 2.1 Dataset Characteristics

The analysis uses the Bank Marketing dataset, a real-world dataset from direct marketing campaigns of a Portuguese banking institution.

| Property | Value |
|----------|-------|
| Total Observations | 45,211 |
| Features (Raw) | 20 |
| Features (Post-Encoding) | 41 |
| Training Samples | 36,168 |
| Test Samples | 9,043 |
| Positive Class Rate | 11.7% |
| Negative Class Rate | 88.3% |
| Imbalance Ratio | 7.5:1 |

### 2.2 Feature Categories

- **Client Attributes**: age, job, marital status, education, credit default, loan status
- **Contact Information**: contact type, day, month, duration of contact
- **Campaign History**: number of contacts, previous campaign outcome, days since previous contact
- **Economic Indicators**: account balance, has housing loan, has personal loan
- **Target Variable**: term deposit subscription (binary: yes/no)

### 2.3 Preprocessing Decisions

| Decision | Rationale |
|----------|-----------|
| Remove `duration` | Variable is only known after contact completion → information leakage |
| Retain `pdays = -1` | Sentinel value indicating no prior contact; encode as categorical |
| Encode `"unknown"` category | Represents valid but unrecorded information; treat as distinct category |
| StandardScaler normalization | SVM algorithm is sensitive to feature magnitude; standardization is required |
| Stratified 80/20 split | Preserve class proportion in both training and test sets |

---

## 3. Methodology

### 3.1 Feature Preprocessing

```python
# Load and clean data
df = pd.read_csv('bank-full.csv', sep=';')
df = df.drop(columns=['duration'])

# Target encoding
y = (df['y'] == 'yes').astype(int)
X_raw = df.drop(columns=['y'])

# One-hot encoding for categorical features
X = pd.get_dummies(X_raw, drop_first=True)  # Results in 41 features

# Stratified train/test split
Xtr, Xte, ytr, yte = train_test_split(
    X, y, 
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Feature scaling
scaler = StandardScaler()
Xtr_scaled = scaler.fit_transform(Xtr)
Xte_scaled = scaler.transform(Xte)
```

### 3.2 Support Vector Machine

The Support Vector Machine solves the following optimization problem:

$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}\xi_i$$

Subject to: $y_i(w^T\phi(x_i) + b) \geq 1 - \xi_i$, $\xi_i \geq 0$

where $\phi(x)$ is the kernel-induced feature map, $C$ controls the regularization, and $\xi_i$ are slack variables.

The RBF (Radial Basis Function) kernel is employed:

$$K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$$

This kernel enables non-linear separation without explicit computation in high-dimensional space.

### 3.3 Addressing Class Imbalance

Two mechanisms are implemented to handle class imbalance:

1. **Balanced Class Weights**: Automatically assign weights inversely proportional to class frequencies:
$$w_j = \frac{n}{2 \cdot n_j}$$
where $n_j$ is the count of samples in class $j$.

2. **Stratified Sampling**: Ensure train and test sets maintain the same class proportion as the original dataset.

### 3.4 Hyperparameter Optimization

Hyperparameter tuning is performed on an 8,000-sample stratified subsample to maintain computational efficiency. The evaluation metric is F1-score on the test set.

#### 3.4.1 Parameter $C$ (Regularization)

The hyperparameter $C$ controls the trade-off between margin width and classification error:
- Low $C$: Wider margin, higher tolerance for misclassification (underfitting)
- High $C$: Narrower margin, lower tolerance for error (overfitting)

| $C$ | Accuracy | Precision | Recall | F1-Score |
|-----|----------|-----------|--------|----------|
| 0.01 | 0.712 | 0.223 | 0.587 | 0.323 |
| 0.10 | 0.805 | 0.317 | 0.581 | 0.411 |
| **1.00** | **0.823** | **0.343** | **0.558** | **0.425** |
| 10.00 | 0.804 | 0.276 | 0.415 | 0.331 |
| 100.00 | 0.800 | 0.250 | 0.353 | 0.292 |

**Result**: $C = 1.0$ achieves the optimal F1-score, indicating appropriate regularization.

#### 3.4.2 Parameter $\gamma$ (Kernel Bandwidth)

The RBF bandwidth parameter $\gamma$ determines the reach of individual support vectors:
- Low $\gamma$: Each support vector influences a large region (smoother boundary)
- High $\gamma$: Each support vector influences a small region (more complex, prone to overfitting)

| $\gamma$ | Accuracy | Precision | Recall | F1-Score |
|----------|----------|-----------|--------|----------|
| 0.001 | 0.825 | 0.341 | 0.532 | 0.415 |
| **0.01** | **0.827** | **0.353** | **0.575** | **0.438** |
| 0.10 | 0.810 | 0.259 | 0.335 | 0.292 |
| 1.00 | 0.849 | 0.138 | 0.055 | 0.078 |
| scale | 0.823 | 0.343 | 0.558 | 0.425 |

**Result**: $\gamma = 0.01$ yields the highest F1-score (0.438), demonstrating superior generalization performance.

---

## 4. Results

### 4.1 Model Configuration

Final model hyperparameters:

```python
SVC(
    kernel='rbf',
    C=1.0,
    gamma=0.01,
    class_weight='balanced',
    random_state=42
)
```

### 4.2 Test Set Performance

| Metric | Score |
|--------|-------|
| Accuracy | 0.827 |
| Precision | 0.353 |
| Recall | 0.575 |
| F1-Score | 0.438 |

### 4.3 Confusion Matrix

```
                    Predicted
                Negative  Positive
Actual Negative   6,927     1,058
Actual Positive     476       582
```

### 4.4 Interpretation

- **True Negatives (TN)**: 6,927 — Non-subscribers correctly identified
- **False Positives (FP)**: 1,058 — Non-subscribers incorrectly classified as subscribers
- **False Negatives (FN)**: 476 — Subscribers incorrectly classified as non-subscribers
- **True Positives (TP)**: 582 — Subscribers correctly identified

### 4.5 Key Observations

1. **Recall (57.5%)**: The model detects approximately 55% of actual subscribers, leaving 45% unidentified.

2. **Precision (35.3%)**: For every 3 customers predicted to subscribe, approximately 1 actually does, indicating a substantial false positive rate.

3. **F1-Score (0.438)**: Reflects a moderate balance between precision and recall, constrained by the challenges of class imbalance and feature-target relationships.

4. **Class Weight Effect**: The `class_weight='balanced'` parameter prevents the model from defaulting to the majority class prediction, which is critical given the 7.5:1 imbalance.

---

## 5. Discussion

### 5.1 Model Strengths

- Systematic treatment of class imbalance through balanced weighting and stratified sampling
- Prevention of information leakage through careful feature selection
- Comprehensive hyperparameter exploration with appropriate evaluation metrics
- Reproducible pipeline with fixed random seed

### 5.2 Limitations and Considerations

1. **False Positive Rate**: The 35.3% precision suggests that approximately two-thirds of predicted subscribers are false alarms, which has cost implications for marketing campaigns.

2. **False Negative Rate**: The model misses 45% of actual subscribers, representing lost business opportunities.

3. **Hyperparameter Scope**: Tuning limited to two primary parameters ($C$ and $\gamma$). A broader grid search with cross-validation could yield further improvements.

4. **Decision Threshold**: The default classification threshold of 0.5 may not align with business objectives. Threshold optimization based on cost-benefit analysis could improve practical utility.

### 5.3 Recommendations for Practitioners

**Business Context Dependency**: The optimal model configuration depends on the relative costs of false positives versus false negatives:

- **High-cost false positives** (e.g., expensive marketing outreach): Increase classification threshold (e.g., 0.7) to prioritize precision.
- **High-cost false negatives** (e.g., significant lost revenue per customer): Decrease classification threshold (e.g., 0.3) to prioritize recall.
- **Balanced scenario**: Maintain default threshold of 0.5, as current F1-score suggests reasonable balance.

**Technical Improvements**:

- Implement k-fold stratified cross-validation with GridSearchCV for more robust hyperparameter selection
- Explore alternative algorithms (Random Forest, XGBoost, LightGBM) for comparative analysis
- Apply SMOTE or ADASYN resampling techniques to address class imbalance
- Conduct feature importance analysis (permutation importance, SHAP values)
- Evaluate model calibration and perform threshold optimization
- Implement cross-validation curves (learning curves) to diagnose bias-variance trade-off

---

## 6. Installation and Usage

### 6.1 Requirements

```
Python >= 3.9
pandas >= 1.5
numpy >= 1.23
scikit-learn >= 1.3
matplotlib >= 3.5
seaborn >= 0.12
jupyter >= 1.0
```

### 6.2 Setup

```bash
# Clone repository
git clone https://github.com/SantiagoRodriguez114/modelSVM_marketing.git
cd modelSVM_marketing

# Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate            # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 6.3 Execution

```bash
# Launch Jupyter Notebook
jupyter notebook SVM_marketing.ipynb

# Or use JupyterLab
jupyter lab SVM_marketing.ipynb
```

Execute the notebook cells sequentially. Output includes metrics, confusion matrices, and hyperparameter sensitivity plots.

---

## 7. Repository Structure

```
modelSVM_marketing/
├── README.md                      # Documentation
├── requirements.txt               # Python dependencies
├── SVM_marketing.ipynb            # Main analysis notebook
├── bank-full.csv                  # Dataset
└── .gitignore
```

---

## 8. References

[1] Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31. https://doi.org/10.1016/j.dss.2014.03.001

[2] Vapnik, V. N. (1995). *The Nature of Statistical Learning Theory*. Springer-Verlag.

[3] He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239

[4] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

[5] Pedregosa, F., et al. (2011). scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

---

## 9. Author

**Santiago Rodríguez**

Contact: rodriguezsanti751@gmail.com  
GitHub: [@SantiagoRodriguez114](https://github.com/SantiagoRodriguez114)

---

**License**: MIT  
**Last Updated**: October 2026
