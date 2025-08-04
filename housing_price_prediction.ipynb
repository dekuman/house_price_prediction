# Predicting House Prices using Machine Learning

# Step 1: Data Loading & Preprocessing

from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Target'] = housing.target

# Explore dataset
print("Feature Descriptions:\n", housing.DESCR)
print("\nDataset Shape:", df.shape)

# Plot histograms
plt.figure(figsize=(15, 10))
df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()

# Optional: Pairplot (uncomment if needed)
# sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Target']])

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(df[housing.feature_names])
y = df['Target']

# Step 2: Model Building

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose Model: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Or: Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred_rf)
mse = mean_squared_error(y_test, y_pred_rf)
r2 = r2_score(y_test, y_pred_rf)

print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)

# Feature Importances
feature_importance = pd.Series(rf_model.feature_importances_, index=housing.feature_names)
feature_importance.sort_values(ascending=False).plot(kind='bar', title='Feature Importances')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# Save model
joblib.dump(rf_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Step 3: Streamlit Deployment (in app.py)
# See app.py file separately

# Step 4: Documentation
# Create README.md or PDF separately including:
# - Project Summary
# - How to run app: streamlit run app.py
# - Metrics Explanation
# - Screenshot of the Streamlit App (when deployed)

print("\nModel and Scaler saved as 'model.pkl' and 'scaler.pkl'")
