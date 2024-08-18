import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics

# Load the dataset
car_dataset = pd.read_csv('car data.csv')

# Verify the data is loaded correctly by printing the first few rows
print(car_dataset.head())

# Check the number of rows and columns
dataset_shape = car_dataset.shape
print("Number of rows and columns:", dataset_shape)

# Get some information about the dataset
car_dataset.info()

# Check the number of missing values
print("Missing values in each column:\n", car_dataset.isnull().sum())

# Check the distribution of categorical data
print("Fuel Type Distribution:\n", car_dataset['Fuel_Type'].value_counts())
print("Seller Type Distribution:\n", car_dataset['Seller_Type'].value_counts())
print("Transmission Distribution:\n", car_dataset['Transmission'].value_counts())

# Encoding categorical columns
car_dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
car_dataset.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
car_dataset.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

# Display the first few rows after encoding
print(car_dataset.head())

# Define the features and target variable
X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = car_dataset['Selling_Price']

# Print the features and target
print("Features (X):\n", X)
print("Target (Y):\n", Y)

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

# Load and train the Linear Regression model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

# Prediction on training data
training_data_prediction = lin_reg_model.predict(X_train)

# Calculate R squared error for training data
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error on Training Data: ", error_score)

# Plot the actual vs predicted prices for training data
plt.scatter(Y_train, training_data_prediction, color='red', label='Predicted Price')
plt.scatter(Y_train, Y_train, color='green', label='Actual Price')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices (Training Data)")
plt.legend()
plt.show()

# Prediction on test data
test_data_prediction = lin_reg_model.predict(X_test)

# Calculate R squared error for test data
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error on Test Data: ", error_score)

# Plot the actual vs predicted prices for test data
plt.scatter(Y_test, test_data_prediction, color='red', label='Predicted Price')
plt.scatter(Y_test, Y_test, color='green', label='Actual Price')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices (Test Data)")
plt.legend()
plt.show()

# Load and train the Lasso Regression model
lass_reg_model = Lasso()
lass_reg_model.fit(X_train, Y_train)

# Prediction on training data using Lasso
training_data_prediction = lass_reg_model.predict(X_train)

# Calculate R squared error for training data (Lasso)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error on Training Data (Lasso): ", error_score)

# Plot the actual vs predicted prices for training data (Lasso)
plt.scatter(Y_train, training_data_prediction, color='red', label='Predicted Price')
plt.scatter(Y_train, Y_train, color='green', label='Actual Price')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices (Training Data) - Lasso")
plt.legend()
plt.show()

# Prediction on test data using Lasso
test_data_prediction = lass_reg_model.predict(X_test)

# Calculate R squared error for test data (Lasso)
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error on Test Data (Lasso): ", error_score)

# Plot the actual vs predicted prices for test data (Lasso)
plt.scatter(Y_test, test_data_prediction, color='red', label='Predicted Price')
plt.scatter(Y_test, Y_test, color='green', label='Actual Price')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices (Test Data) - Lasso")
plt.legend()
plt.show()
