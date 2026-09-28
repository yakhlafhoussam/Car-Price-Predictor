# 🚗 Used Car Price Prediction

A Machine Learning project for predicting the selling price of used cars based on their characteristics.

The project covers the complete Machine Learning workflow, from data cleaning and exploratory analysis to model training, hyperparameter tuning, evaluation, comparison, and deployment through a Streamlit application.

---

## 📌 Project Overview

The objective of this project is to build a regression model capable of estimating the selling price of a used car using information such as:

* Car name
* Manufacturing year
* Kilometers driven
* Fuel type
* Seller type
* Transmission
* Owner type

The project focuses not only on obtaining good prediction performance, but also on understanding the dataset, handling missing values and outliers, comparing different algorithms, and selecting a final production model.

---

## 📊 Dataset

The dataset contains used-car listings with the following columns:

| Column          | Description                     |
| --------------- | ------------------------------- |
| `name`          | Car model/name                  |
| `year`          | Manufacturing year              |
| `selling_price` | Selling price — target variable |
| `km_driven`     | Number of kilometers driven     |
| `fuel`          | Fuel type                       |
| `seller_type`   | Type of seller                  |
| `transmission`  | Transmission type               |
| `owner`         | Previous owner information      |

### Dataset size

Original dataset:

```text
4340 rows × 8 columns
```

After removing duplicate records:

```text
3705 rows × 8 columns
```

---

## 🔎 Project Workflow

The project follows these main steps:

### 1. Data Understanding

* Load the dataset
* Inspect dimensions and data types
* Analyze categorical and numerical features
* Identify missing values
* Detect duplicated records
* Generate descriptive statistics

### 2. Data Cleaning

Several preprocessing operations were performed:

* Removing duplicate rows
* Handling missing values
* Converting numerical columns to appropriate types
* Imputing missing values using contextual/group-based strategies
* Checking inconsistent values

For example, missing values were handled using information from related features such as:

```text
name
year
fuel
seller_type
owner
```

rather than relying only on global statistics.

### 3. Exploratory Data Analysis

The dataset was analyzed using different visualizations:

* Histograms
* Box plots
* Scatter plots
* Correlation analysis
* Pair plots
* Target distribution analysis

These visualizations helped identify relationships between car characteristics and selling price.

### 4. Outlier Detection

Outliers were investigated using techniques such as:

* IQR
* Z-score
* Box plots

Special attention was given to extreme values in:

```text
selling_price
km_driven
```

The goal was to distinguish genuine high-value vehicles from potentially problematic observations.

---

## 🤖 Machine Learning Models

Several regression algorithms were trained and evaluated.

The project includes models such as:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting
* XGBoost

Each model was trained using a preprocessing pipeline to ensure consistent data transformation.

---

## ⚙️ Hyperparameter Tuning

For the most promising models, hyperparameter optimization was performed using `GridSearchCV`.

### Random Forest

Parameters explored included:

```python
{
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10]
}
```

### XGBoost

XGBoost hyperparameters were also optimized to improve the model's performance and generalization.

Cross-validation was used during the tuning process to obtain a more reliable estimate of model performance.

---

## 📏 Model Evaluation

The models were evaluated using three main regression metrics.

### RMSE — Root Mean Squared Error

Measures the average prediction error while giving more weight to large errors.

```text
RMSE = √MSE
```

Lower values indicate smaller prediction errors.

### MAE — Mean Absolute Error

Measures the average absolute difference between predicted and actual prices.

```text
MAE = mean(|y_true - y_pred|)
```

Lower values are better.

### R² — Coefficient of Determination

Measures how much of the variation in the target variable is explained by the model.

```text
R² = 1 - SSres / SStot
```

A value closer to `1` indicates that the model explains more of the observed variation.

---

## 📈 Model Comparison

The trained models are compared using:

* RMSE
* MAE
* R²
* Prediction vs. actual plots
* Residual analysis

Example comparison structure:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>RMSE</th>
      <th>MAE</th>
      <th>R²</th>
      <th>R² (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear Regression</td>
      <td>416919.836243</td>
      <td>245797.457538</td>
      <td>0.221984</td>
      <td>22.198401</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Random Forest</td>
      <td>253428.836169</td>
      <td>135459.842971</td>
      <td>0.712528</td>
      <td>71.252806</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XGBoost</td>
      <td>195290.554733</td>
      <td>112728.500000</td>
      <td>0.829295</td>
      <td>82.929516</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SVR</td>
      <td>492122.249271</td>
      <td>285958.914003</td>
      <td>-0.084000</td>
      <td>-8.400031</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Random Forest - After</td>
      <td>227792.860232</td>
      <td>118951.365986</td>
      <td>0.767746</td>
      <td>76.774578</td>
    </tr>
    <tr>
      <th>5</th>
      <td>XGBoost - After</td>
      <td>188476.794540</td>
      <td>106094.617188</td>
      <td>0.840999</td>
      <td>84.099925</td>
    </tr>
  </tbody>
</table>
</div>

> The final values are generated from the experiments in the notebook.

---

## 📉 Prediction Analysis

The project also uses prediction-vs-actual visualizations to understand model behavior.

A good regression model should produce predictions close to the diagonal:

```text
Predicted Price
       │
       │       /
       │     /
       │   /
       │ /
       └──────────── Actual Price
```

Residual plots are also used to identify systematic errors and unusual predictions.

---

## 🏆 Final Model

After comparing the different models and tuning the strongest candidates, the selected model is saved for later use by the application.

The trained model is serialized using `joblib`.

Example:

```python
joblib.dump(model, "models/xgboost_final.pkl")
```

The application then loads the trained model instead of retraining it every time.

---

## 🌐 Streamlit Application

A Streamlit application is included to provide an interactive interface for price prediction.

The user can enter information such as:

* Car model
* Manufacturing year
* Kilometers driven
* Fuel type
* Seller type
* Transmission
* Owner type

The application processes the input and returns an estimated selling price.

### Run the application

From the project root:

```bash
streamlit run app/app.py
```

---

## 📁 Project Structure

```text
Car-Price-Predictor/
│
├── app/
│   └── app.py
│
├── data/
│   └── car-price.csv 
│
├── models/
│   └── xgboost_final.pkl
│
├── notebooks/
│   └── exploration.ipynb
│
├── requirements.txt
│
└── README.md
```

The exact model files may vary depending on the final experiment.

---

## 🛠️ Technologies

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost

### Model Serialization

* Joblib

### Application

* Streamlit

### Development

* Jupyter Notebook
* VS Code
* Python Virtual Environment

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yakhlafhoussam/Car-Price-Predictor.git
cd brief-2
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Project

### Run the notebook

Open the notebook using Jupyter or VS Code and execute the cells in order.

### Run the Streamlit application

```bash
streamlit run app/app.py
```

---

## 🎯 Project Objectives

This project was developed to practice the complete Machine Learning workflow:

* Understand and clean real-world data
* Handle missing values
* Detect and analyze outliers
* Perform exploratory data analysis
* Prepare features for Machine Learning
* Train multiple regression models
* Optimize model hyperparameters
* Compare model performance
* Analyze prediction errors
* Save a production-ready model
* Build an interactive prediction application

---

## 🔮 Possible Improvements

Future improvements could include:

* Collecting a larger and more recent dataset
* Adding additional vehicle features
* Advanced feature engineering
* More extensive hyperparameter optimization
* Cross-validation analysis
* Explainable AI using SHAP
* Model monitoring
* Deploying the Streamlit application online
* Adding prediction confidence or price ranges

---

## 👨‍💻 Author

**Houssam YAKHLAF**

Machine Learning / Full Stack Web Development Student

Morocco

---

## 📄 License

This project was developed for educational purposes as part of a Machine Learning project.
