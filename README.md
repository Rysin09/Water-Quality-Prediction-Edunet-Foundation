# Water Quality Prediction using Machine Learning

### AICTE Virtual Internship Project | Sponsored by Shell | June 2025

---

## Overview

Access to clean water is a critical global concern. Accurate and timely prediction of water quality metrics plays a significant role in pollution detection, environmental protection, and health safety. 

This project utilizes **supervised machine learning** to predict multiple water quality parameters simultaneously using a **MultiOutputRegressor** wrapped around a **RandomForestRegressor**.

---

## Problem Statement

Given a dataset containing various water sample features, the objective is to predict a set of dependent water quality parameters that indicate the level of pollution in the water body.

---

## Goals

- Collect and preprocess real-world water quality datasets
- Use regression-based machine learning for multi-target prediction
- Build and fine-tune a predictive pipeline
- Evaluate performance using regression metrics

---

## Technologies and Tools

- **Language:** Python 3.12  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Visualization:** Matplotlib, Seaborn  
- **Environment:** Jupyter Notebook  

---

## Machine Learning Approach

- Model: `RandomForestRegressor` wrapped with `MultiOutputRegressor`
- Task Type: Multi-target Regression
- Pipeline: Data Preprocessing → Model Training → Evaluation

---

## Predicted Water Quality Parameters

The model is trained to predict the following:

- NH₄ (Ammonium)
- BOD₅ (Biochemical Oxygen Demand over 5 days)
- Colloids
- O₂ (Dissolved Oxygen)
- NO₃ (Nitrate)
- NO₂ (Nitrite)
- SO₄ (Sulfate)
- PO₄ (Phosphate)
- CL (Chloride)

---

## Model Evaluation

The model's performance was evaluated using:

- **R² Score**
- **Mean Squared Error (MSE)**

The results indicate consistent and acceptable performance across all target variables.

---

## Sample Workflow

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the model
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
