import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("optimized_medical_treatment_costs.csv")

# Check column names (For debugging)
print("Columns in dataset:", data.columns)

# Apply Log Transformation to normalize target variable
data['Estimated Cost'] = np.log1p(data['Estimated Cost'])  # log(1 + x) to prevent log(0) issues

# Separate Features and Target Variable
X = data.drop(columns=['Estimated Cost'])
y = data['Estimated Cost']

# Identify Categorical and Numerical Features
categorical_features = ['Health Condition', 'Treatment Type']
numerical_features = ['Age']

# Preprocessing Pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),  # Scale numerical features
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Encode categorical features
])

# Model Pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())  
])

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Reverse Log Transformation for Meaningful Comparison
y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test_original, y_pred_original)
print("Optimized Mean Squared Error (MSE) with Linear Regression:", mse)

# Show Predictions
results = X_test.copy()
results['Actual Cost'] = y_test_original
results['Predicted Cost'] = y_pred_original
print(results.head())

# Save the trained model
with open("linear_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained & saved as 'linear_model.pkl'")

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))  # Increase figure size for better readability
plt.scatter(y_test_original, y_pred_original, alpha=0.6, color='blue', label='Predicted vs Actual')  # Add transparency for better visualization
plt.plot([min(y_test_original), max(y_test_original)], [min(y_test_original), max(y_test_original)], color='red', linestyle='--', label='Ideal Line (y = x)')  # Add ideal line for reference

# Add labels and title
plt.xlabel('Actual Estimated Cost (USD)', fontsize=12)
plt.ylabel('Predicted Estimated Cost (USD)', fontsize=12)
plt.title('Actual vs Predicted Estimated Medical Treatment Costs', fontsize=14, fontweight='bold')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)
# Add legend
plt.legend()
# Show the plot
plt.show()