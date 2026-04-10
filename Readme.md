### Titanic Survival Classification

## Project Overview  
This project uses machine learning to predict whether a passenger survived the Titanic disaster.  
A Logistic Regression model is trained using passenger information such as age, gender, ticket class, and fare.

This is a classic binary classification problem and is commonly used for learning data preprocessing and model evaluation.

---

## Objective  
To build a classification model that predicts:

- 1 → Survived  
- 0 → Did Not Survive  

---

## Dataset  
The dataset used is the Titanic dataset, loaded using the seaborn library.

---

## Features Used  
- pclass – Ticket class (1st, 2nd, 3rd)  
- sex – Gender of passenger  
- age – Age of passenger  
- fare – Ticket fare  

---

## Target Variable  
- survived  

---

## Data Preprocessing  
- Selected relevant columns only  
- Filled missing values in age and fare using median  
- Encoded categorical feature (sex) using Label Encoding  
- Split data into training and testing sets (80/20)  

---

## Machine Learning Model  
- Algorithm: Logistic Regression  
- Reason: Simple, efficient, and effective for binary classification tasks  

---

## Model Evaluation  
The model is evaluated using:

- Accuracy Score  
- Classification Report (Precision, Recall, F1-score)  
- Confusion Matrix (visualized using heatmap)  

---

## Result  
- Achieved approximately 80% accuracy  
- Gender and ticket class were found to be strong predictors of survival  

---

## Technologies Used  
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)  
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)  
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)  
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)  
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white)  
![Seaborn](https://img.shields.io/badge/Seaborn-0A7D7E?style=flat)  
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

## Project Structure  
titanic-survival-classification/
├── notebooks/
│ └── Titanic_Survival_Classification.ipynb
├── data/
│ └── titanic.csv
├── requirements.txt
├── README.md
└── LICENSE
---

## How to Run  
1. Clone the repository  
2. Open `Titanic_Survival_Classification.ipynb`  
3. Run all cells in Jupyter Notebook  

---

## Conclusion  
This project demonstrates the complete machine learning workflow, from data cleaning to model evaluation, using a real-world dataset.

---

## Author  
Hassan Ali  
Data Science and Machine Learning Enthusiast  
