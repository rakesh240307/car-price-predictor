# 🚗 Car Price Predictor

An end-to-end Machine Learning web application that predicts the estimated selling price of a used car based on its specifications.
Built using **XGBoost Regression** and **Streamlit**, with the trained model deployed as a live web application.

## 📌 Project Overview

Car Price Predictor estimates the selling price of a used car based on its specifications.

The application takes vehicle details such as brand, model, age, kilometers driven, mileage, engine, power, fuel type, transmission, seller type, and seats, then uses a trained XGBoost model to generate the estimated price.

## ✨ Features

- Used car price prediction
- Brand and model selection
- Vehicle specification inputs
- XGBoost-based prediction
- Automated data preprocessing
- Interactive Streamlit interface
- Saved model and preprocessing pipeline
- Live cloud deployment

## 📂 Dataset & Data Preparation

The dataset contains used-car specifications and selling prices.

- Final dataset: **15,244 records**
- Duplicate records were removed.
- Redundant `car_name` feature was removed.
- Target variable: `selling_price`
- Train/Test split: **80/20**
- Numerical features were scaled using `StandardScaler`.
- Categorical features were encoded using `OneHotEncoder`.
- `handle_unknown="ignore"` was used for unseen categories.
- Preprocessing was fitted only on training data to avoid data leakage.

## 🤖 Models & Results

Three regression models were evaluated:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | ₹168,925 | ₹345,727 | 0.8034 |
| Random Forest | ₹99,899 | ₹267,122 | 0.8826 |
| **XGBoost** | **₹93,254** | **₹189,168** | **0.9411** |

### 🏆 Final Model

**XGBoost Regressor** was used for the deployed application.

- **R²:** 0.9411
- **MAE:** ₹93,254
- **RMSE:** ₹189,168

## 🖥️ Application

The application is built with **Streamlit** and provides an interactive interface for entering vehicle details and generating a price estimate.

### Prediction Flow

User Input → Preprocessing → XGBoost Model → Estimated Price

The trained model and preprocessing pipeline are saved using **Joblib** and loaded by the application without retraining.

## 🛠️ Tech Stack

- **Python** — Programming
- **Pandas & NumPy** — Data processing
- **Scikit-learn** — Preprocessing and evaluation
- **XGBoost** — Machine learning model
- **Joblib** — Model persistence
- **Streamlit** — Web application
- **Git & GitHub** — Version control
- **Streamlit Community Cloud** — Deployment

## 📁 Project Structure

```text
car-price-predictor/
├── dataset/
├── models/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

## ⚙️ Run Locally

```bash
git clone https://github.com/rakesh240307/car-price-predictor.git
cd car-price-predictor
pip install -r requirements.txt
streamlit run app.py

## ⚠️ Limitations & 🔮 Future Improvements

Predictions may vary because factors such as vehicle condition, location, service history, ownership, and current market demand are not included in the dataset.

Future improvements include adding these factors, using newer market data, adding SHAP-based explanations, and providing prediction ranges.

## 👨‍💻 Author

**Rakesh Singh**  
B.Tech Computer Science Engineering

🔗 [GitHub](https://github.com/rakesh240307)

## 🚀 Live Demo

👉 **[Try the Car Price Predictor](https://car-price-predictor-hrxtnt5r7v9nye84c68c9b.streamlit.app/)**
  


