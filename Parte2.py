import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the data into a pandas DataFrame
df = pd.read_csv("baseball.csv")
print(df)
# Replace the comma in the number with an empty string and convert to float
df = df.replace(["']", ",", ":"], "", regex=True).astype(float)

# Split the data into input features (X) and target variable (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Fit the linear regression model to the data
reg = LinearRegression().fit(X, y)

# Obtain the R2 score of the model
y_pred = reg.predict(X)
r2 = r2_score(y, y_pred)
print("R2 score:", r2)

# Obtain the constants (intercept and coefficients) of the model
intercept = reg.intercept_
coeffs = reg.coef_

# Express the equation as a string
equation = "y = "
equation += str(intercept) + " + "
for i, coeff in enumerate(coeffs):
    equation += str(coeff) + " * X" + str(i) + " + "
equation = equation[:-3]
print("Equation:", equation)

# Predict the number of attendees given X and Y teams, day of the week, time, and state
X_new = np.array([[X_team, Y_team, day_of_week, time, state]])
attendance_prediction = reg.predict(X_new)
print("Attendance prediction:", attendance_prediction[0])
