<div align="center">

# 🚢 Titanic Survival Classification

### Predicting who survives — with machine learning.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-22c55e?style=for-the-badge)]()

<br/>

> *"On April 15, 1912, the RMS Titanic sank after colliding with an iceberg. 1,502 out of 2,224 passengers and crew perished. But survival wasn't random — it followed a pattern. This project finds it."*

<br/>

---

</div>

## 📌 Table of Contents

- [Overview](#-overview)
- [The Problem](#-the-problem)
- [Dataset](#-dataset)
- [Feature Engineering](#-feature-engineering)
- [Methodology](#-methodology)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Key Insights](#-key-insights)
- [Tech Stack](#-tech-stack)
- [Author](#-author)

---

## 🔍 Overview

This project tackles the **Titanic Survival Classification** problem — one of the most iconic datasets in data science. Using **Logistic Regression**, we build an end-to-end machine learning pipeline that predicts whether a passenger survived the disaster based on their personal and ticketing information.

The project covers the **complete ML workflow**:

```
Raw Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Insights
```

**Final Model Accuracy: ~80%** ✅

---

## 🎯 The Problem

| Type | Detail |
|------|--------|
| **Task** | Binary Classification |
| **Target** | `survived` — `1` (Survived) or `0` (Did Not Survive) |
| **Challenge** | Missing values, categorical encoding, class imbalance |
| **Metric** | Accuracy, Precision, Recall, F1-Score |

---

## 📊 Dataset

- **Source:** Seaborn's built-in Titanic dataset (`seaborn.load_dataset('titanic')`)
- **Size:** 891 rows × 15 columns (original)
- **After preprocessing:** 4 features selected, missing values handled

| Column | Description | Type |
|--------|-------------|------|
| `pclass` | Ticket class (1st, 2nd, 3rd) | Ordinal |
| `sex` | Gender of passenger | Categorical |
| `age` | Age in years | Continuous |
| `fare` | Ticket fare paid (£) | Continuous |
| `survived` ⭐ | Survival outcome (target) | Binary |

---

## 🛠 Feature Engineering

```python
# Step 1 — Select relevant columns
features = ['pclass', 'sex', 'age', 'fare', 'survived']
df = titanic[features].copy()

# Step 2 — Handle missing values with median imputation
df['age'].fillna(df['age'].median(), inplace=True)
df['fare'].fillna(df['fare'].median(), inplace=True)

# Step 3 — Encode categorical feature
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])  # male=1, female=0

# Step 4 — Train/Test split (80/20)
X = df.drop('survived', axis=1)
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 🔬 Methodology

### Why Logistic Regression?

Logistic Regression was chosen as the baseline model because:

- ✅ **Interpretable** — coefficients directly reveal feature impact
- ✅ **Fast to train** — ideal for small-to-medium tabular datasets
- ✅ **No scaling required** for this use case
- ✅ **Probabilistic output** — gives survival *probability*, not just a binary label
- ✅ **Strong baseline** — often competitive with complex models on well-structured data

### Model Pipeline

```
1. Load Data          →  seaborn.load_dataset('titanic')
2. Select Features    →  pclass, sex, age, fare
3. Impute Missing     →  Median for age and fare
4. Encode Sex         →  LabelEncoder (male=1, female=0)
5. Split Dataset      →  80% train / 20% test
6. Train Model        →  LogisticRegression(max_iter=200)
7. Evaluate           →  Accuracy, Classification Report, Confusion Matrix
```

---

## 📈 Results

### Model Performance

| Metric | Class 0 (Did Not Survive) | Class 1 (Survived) | Overall |
|--------|--------------------------|-------------------|---------|
| **Precision** | 0.83 | 0.76 | — |
| **Recall** | 0.85 | 0.73 | — |
| **F1-Score** | 0.84 | 0.74 | — |
| **Accuracy** | — | — | **~80%** |

### Confusion Matrix

```
Predicted →        Not Survived    Survived
Actual Not Survived  [  TN ✅  ]  [  FP ❌  ]
Actual Survived      [  FN ❌  ]  [  TP ✅  ]
```

> The heatmap generated in the notebook provides a visual breakdown of correct vs. incorrect predictions.

---

## 📁 Project Structure

```
titanic-survival-classification/
│
├── 📓 notebooks/
│   └── Titanic_Survival_Classification.ipynb   ← Main notebook (run this)
│
├── 📂 data/
│   └── titanic.csv                              ← Dataset (auto-loaded via seaborn)
│
├── 📄 requirements.txt                          ← All dependencies
├── 📄 README.md                                 ← You are here
└── 📄 LICENSE                                   ← MIT License
```

---

## 🚀 How to Run

### Prerequisites

Make sure you have Python 3.10+ and pip installed.

### 1. Clone the Repository

```bash
git clone https://github.com/hassan-ali786/titanic-survival-classification.git
cd titanic-survival-classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook notebooks/Titanic_Survival_Classification.ipynb
```

### 4. Run All Cells

From the menu: **Kernel → Restart & Run All**

> 💡 No need to manually download the dataset — it's loaded automatically via `seaborn.load_dataset('titanic')`.

---

## 💡 Key Insights

After training and evaluating the model, several patterns emerged from the data:

- **Gender was the strongest predictor** — female passengers had a significantly higher survival rate, consistent with the "women and children first" evacuation protocol
- **Ticket class mattered** — 1st class passengers survived at much higher rates than 3rd class
- **Age showed modest influence** — younger passengers (especially children) were slightly more likely to survive
- **Fare correlated with survival** — passengers who paid higher fares (often 1st class) survived more, but this partially overlaps with `pclass`

These findings align with historical accounts of the Titanic disaster.

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Core programming language |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | Data loading and manipulation |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Numerical operations |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | ML model, encoding, evaluation |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat) | Plotting |
| ![Seaborn](https://img.shields.io/badge/Seaborn-0A7D7E?style=flat) | Dataset loading, heatmaps |
| ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) | Interactive development environment |

---

## 🔮 Future Improvements

- [ ] Try advanced models: Random Forest, XGBoost, LightGBM
- [ ] Add feature: `family_size` (SibSp + Parch + 1)
- [ ] Extract title from `name` column (Mr., Mrs., Miss., Master.)
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Deploy as a web app using Streamlit or Flask

---

## 👤 Author

<div align="center">

**Hassan Ali**
*Data Scientist | ML Engineer*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/hassanali)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com/hassanali789)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/hassan-ali786)

</div>

---

<div align="center">

⭐ **If this project helped you, please give it a star!** ⭐

*Made with curiosity, data, and a lot of Python*

</div>
