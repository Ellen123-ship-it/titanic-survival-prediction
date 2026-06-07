# titanic-survival-prediction
Predicting Titanic survival using Python, ML, and a Dash dashboard

## Overview
This project explores the question:

**Can we predict whether a passenger survived the Titanic disaster based on demographic and ticket information?**

Using the Titanic dataset, the project demonstrates a complete data science workflow:

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering
- Logistic Regression modeling
- Model evaluation
- An interactive Plotly Dash dashboard for visualization and prediction

---

## Project Structure
titanic-survival-prediction/
- eda_and_model.py — Data loading, EDA, cleaning, model training & evaluation  
- dashboard.py — Plotly Dash app for visualizations and survival prediction  
- requirements.txt — Python dependencies  
- README.md — Project documentation  

---

## Dataset
The project uses the Seaborn Titanic dataset, containing:

- Age  
- Sex  
- Passenger class  
- Fare  
- Siblings/spouses aboard (sibsp)  
- Parents/children aboard (parch)  
- Embarkation port  
- Survival outcome (0 = died, 1 = survived)

---

## Methods

### 1. Exploratory Data Analysis (EDA)
- Survival distribution  
- Age distribution  
- Passenger class distribution  
- Correlation heatmap  
- Missing value analysis  

### 2. Data Cleaning & Feature Engineering
- Fill missing `age` with median  
- Fill missing `embarked` with mode  
- Drop columns that leak or duplicate survival information:
  - deck, embark_town, alive, who, adult_male, alone  
- One‑hot encode categorical variables using `pd.get_dummies(..., drop_first=True)`

### 3. Model
- Logistic Regression  
- 80/20 train‑test split  
- Metrics: accuracy + confusion matrix  

Typical accuracy: **0.78–0.82**

---

## Dashboard
The interactive dashboard includes:

### Visualizations
- Survival distribution  
- Passenger class distribution  
- Age distribution  
- Confusion matrix  

### Interactive Prediction Tool
Users can adjust:
- Sex  
- Age  
- Passenger class  
- Fare  
- Siblings/spouses aboard  
- Parents/children aboard  
- Embarkation port  

The dashboard returns an estimated **survival probability**.

---

## How to Run

### 1. Install dependencies

### 2. Run the dashboard

Open the link shown in the terminal (usually http://127.0.0.1:8050/).

---

## Interpretation & Conclusion
The model and dashboard show that survival probability is strongly influenced by:

- Sex (female passengers had higher survival rates)  
- Passenger class (1st class had better outcomes)  
- Fare (higher fares correlate with higher survival)  
- Age (younger passengers had slightly higher survival)

This project demonstrates:
- End‑to‑end data science workflow  
- Proper handling of data leakage  
- Building and evaluating a machine learning model  
- Deploying an interactive dashboard for exploration and communication of results  

