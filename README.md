
# 🚗 Car Price Predictor

A machine learning web application that predicts the estimated selling price of a used car based on its specifications.

The project uses **XGBoost Regression** for price prediction and **Streamlit** for the interactive web interface.

---

## 📌 Project Overview

Used-car prices depend on many factors such as:

- Brand
- Model
- Vehicle age
- Kilometers driven
- Fuel type
- Transmission
- Mileage
- Engine capacity
- Maximum power
- Number of seats
- Seller type

This project uses historical used-car data to learn these relationships and provide an estimated selling price.

---

## 🛠️ Tech Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost

### Frontend

- Streamlit

### Model Storage

- Joblib

### Development Environment

- Google Colab
- Google Drive

---

## 🤖 Machine Learning Pipeline

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Duplicate Removal
     ↓
Feature Selection
     ↓
Train/Test Split
     ↓
Data Preprocessing
     ↓
One-Hot Encoding
     ↓
Feature Scaling
     ↓
XGBoost Regression
     ↓
Model Evaluation
     ↓
Streamlit Application
