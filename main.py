import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
import plotly.express as px

# Load data
data = pd.read_csv('car_purchasing.csv')

# Data Preprocessing
# (Include your preprocessing steps here)

# Model Training
X = data.drop('car purchase amount', axis=1)
y = data['car purchase amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
rf = RandomForestRegressor(n_estimators=100)
xgb = XGBRegressor(n_estimators=100)

# Train models
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Predictions
rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)

# Evaluate models
def evaluate_model(name, pred, y_true):
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    r2 = r2_score(y_true, pred)
    print(f"{name} Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}\n")

evaluate_model("Random Forest", rf_pred, y_test)
evaluate_model("XGBoost", xgb_pred, y_test)

# Hyperparameter tuning (example)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10]
}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# After grid search, you can evaluate or use the best model.
