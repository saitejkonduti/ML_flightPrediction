# Flight Price Prediction using Machine Learning  
<img width="1536" height="1024" alt="Flighprediction" src="https://github.com/user-attachments/assets/e051bb6b-091f-4ea1-99ac-721fce9f699c" />

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pandas](https://img.shields.io/badge/Library-Pandas-yellow)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![XGBoost](https://img.shields.io/badge/Algorithm-XGBoost-green)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

##  Overview  

This project aims to **predict flight ticket prices** based on multiple influencing features such as airline, source, destination, total stops, and duration.  
Using **machine learning regression algorithms**, the model learns from historical flight data to estimate future ticket prices.  

The project helps:
- **Travelers** make smarter booking decisions  
- **Agencies** understand pricing trends  
- **Airlines** optimize dynamic pricing strategies  

---

## Objective  

- Build a regression model to predict flight prices accurately.  
- Identify major factors influencing flight prices.  
- Apply **hyperparameter tuning** to improve model performance.  
- Generate a submission-ready output for test data.

---

## Dataset Information  

| File | Description |
|------|--------------|
| `Data_Train.xlsx` | Contains training flight data with attributes like airline, source, destination, stops, duration, and price. |
| `Test_set.xlsx` | Dataset for which predictions are to be made. |
| `Sample_submission.xlsx` | Example format for submitting results. |

---

## Key Features of the Project  

- Cleaned and transformed unstructured flight data  
- Extracted and engineered features such as journey date, departure time, arrival time, and duration  
- Encoded categorical features using **One-Hot** and **Label Encoding**  
- Applied **Random Forest** and **XGBoost Regressors** for prediction  
- Optimized parameters using **GridSearchCV and RandomizedSearchCV** for hyperparameter tuning  
- Evaluated performance using **R² Score**, **MAE**, and **RMSE**

---

## Workflow  

### Data Preprocessing  
- Removed missing values  
- Converted `Date_of_Journey`, `Dep_Time`, and `Duration` into numeric values  
- Extracted **day** and **month** from journey date  

### Feature Engineering  
- Created new columns for total stops, time differences, and encoded airlines  
- Normalized numeric features for better model convergence  

### Model Building  
- Split data into **training and validation sets** (80:20 ratio)  
- Trained multiple ML models:
  - RandomForestRegressor  
  - XGBoostRegressor  
  - GradientBoostingRegressor  

### Hyperparameter Tuning  
Used **GridSearchCV** and **RandomizedSearchCV** to optimize model parameters such as:
- `n_estimators` (number of trees)  
- `max_depth` (tree depth)  
- `learning_rate` (for boosting models)  
- `min_samples_split` and `min_samples_leaf`  

Example:
```python
from sklearn.model_selection import GridSearchCV
grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
}
rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator=rf, param_grid=grid, cv=5, scoring='r2', verbose=2)
grid_search.fit(X_train, y_train)


