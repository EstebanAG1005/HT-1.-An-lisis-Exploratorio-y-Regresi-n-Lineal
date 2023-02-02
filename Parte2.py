import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Se carga la data
df = pd.read_csv("baseball.csv")

# Remove comma from 'attendance' column
df["attendance"] = (
    df["attendance"].str.replace(",", "").str.replace("'", "").str.replace("]", "")
)

# Remove quotes from 'other_info_string' column
df["other_info_string"] = df["other_info_string"].str.replace('"', "")

# Remove quotes from 'other_info_string' column
df["venue"] = df["venue"].str.replace(":", "")

# Remove quotes from 'start_time' column
df["start_time"] = df["start_time"].str.replace('"', "")

# Remove any remaining HTML tags
df["other_info_string"] = df["other_info_string"].str.replace("<.*?>", "")

print(df)

df.to_csv("New.csv", index=False)


# # Importamos el conjunto de datos
# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values

# # Ajustar el modelo de regresión lineal a los datos que tenemos
# reg = LinearRegression().fit(X, y)

# # Calculo del parametro R2
# y_pred = reg.predict(X)
# r2 = r2_score(y, y_pred)
# print("R2 score:", r2)

# # Obtener las constantes del modelo
# intercept = reg.intercept_
# coeffs = reg.coef_

# # Explresar la ecuacion como un string
# equation = "y = "
# equation += str(intercept) + " + "
# for i, coeff in enumerate(coeffs):
#     equation += str(coeff) + " * X" + str(i) + " + "
# equation = equation[:-3]
# print("Equation:", equation)

# # Predict the number of attendees given X and Y teams, day of the week, time, and state
# # Predecir el numero de personas que atenderan al partido en base a X y Y equipos, dia de la semana, tiempo, y estado
# X_new = np.array(
#     [["New York Mets", "Philadelphia Phillies", "Sunday", "7:3p.m", "New York"]]
# )
# attendance_prediction = reg.predict(X_new)
# print("Attendance prediction:", attendance_prediction[0])
