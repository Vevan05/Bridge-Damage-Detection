# 🏗️ Bridge Condition Prediction using Machine Learning

## 📌 Project Overview

This project builds a machine learning model to predict the **condition of bridges** based on structural and operational features such as age, material type, traffic volume, and maintenance level.

The pipeline includes:

* Data preprocessing
* Feature engineering
* Model training using Random Forest
* Hyperparameter tuning with GridSearchCV
* Model evaluation and saving for deployment


## 📊 Dataset Description

| Feature Name      | Description                        |
| ----------------- | ---------------------------------- |
| Age_of_Bridge     | Age of the bridge (in years)       |
| Material_Type     | Type of material (Steel, Concrete) |
| Traffic_Volume    | Number of vehicles per day         |
| Maintenance_Level | Maintenance frequency/type         |
| Bridge_Condition  | Target (0 = Poor, 1 = Good)        |

## ⚙️ Preprocessing Steps

1. **Handling Missing Values**

   * Numerical: Median imputation
   * Categorical: Most frequent value

2. **Encoding Categorical Variables**

   * One-hot encoding for `Material_Type` and `Maintenance_Level`

3. **Feature Scaling**

   * StandardScaler applied to numerical features

4. **Pipeline Integration**

   * All preprocessing steps are combined using `ColumnTransformer`

## 🌲 Model Used

* **Random Forest Classifier**

  * Handles non-linear relationships
  * Robust to noise and overfitting
  * Works well with mixed data types


## 🔍 Hyperparameter Tuning

GridSearchCV was used with 5-fold cross-validation to optimize:

* `n_estimators`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `class_weight`


## 🚀 How to Run the Project

### 1. Install Dependencies

```bash
pip install pandas scikit-learn joblib
```

### 2. Run the Training Script

```bash
python train.py
```


## 💾 Saving the Model

The best model is saved using:

```python
joblib.dump(grid_search.best_estimator_, "best_model.pkl")
```

This file includes:

* Preprocessing pipeline
* Trained Random Forest model


## 📥 Loading and Using the Model

```python
import joblib

model = joblib.load("best_model.pkl")

# Predict on new data
predictions = model.predict(new_data)
```


## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix


## 🧠 Key Features

* End-to-end ML pipeline
* No data leakage (proper train-test split)
* Automated preprocessing
* Hyperparameter optimization
* Ready for deployment


## 🔧 Future Improvements

* Add SMOTE for class imbalance
* Deploy using Flask or FastAPI
* Build a web interface for predictions
