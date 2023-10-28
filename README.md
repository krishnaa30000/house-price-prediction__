# house-price-prediction__
House price prediction using Linear regression
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load your dataset, assuming you have a CSV file
data = pd.read_csv('house_data.csv')

# Define the features (independent variables) and the target variable (house prices)
X = data[['Feature1', 'Feature2', 'Feature3']]  # Add relevant feature columns
y = data['Price']  # Price is your target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared (R2) Score:", r2)

# You can also use the trained model for predictions on new data
# For example, if you want to predict the price of a new house:
new_house_features = np.array([[value1, value2, value3]])  # Replace with actual feature values
predicted_price = model.predict(new_house_features)
print("Predicted Price:", predicted_price)
