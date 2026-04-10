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
Python  
Pandas  
NumPy  
Seaborn  
Matplotlib  
Scikit-learn  
Jupyter Notebook  

---

## Project Structure  
Titanic-Survival-Classification/
│── Titanic_Survival_Classification.ipynb
│── README.md


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
