import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas_profiling


# Se carga la data
df = pd.read_csv("baseball.csv")


# Sirve para eliminar los datos que sean diferentes a numericos y limpia el campo
df["attendance"] = (
    df["attendance"]
    .str.replace(",", "", regex=False)
    .str.replace("'", "", regex=False)
    .str.replace("]", "", regex=False)
)


df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")
df = df.dropna(subset=["attendance"])

# Eliminar la columna de other_info_string
df = df.drop("other_info_string", axis=1)

# Eliminar la columna de other_info_string
df = df.drop("boxscore_url", axis=1)

# Eliminar la columna de other_info_string
df = df.drop("field_type", axis=1)

# Cambiar : por un espacio vaio para limpiar la data
df["venue"] = df["venue"].str.replace(":", "")

df["game_duration"] = df["game_duration"].str.replace(":", "")

# Remover las "" de la columna 'start_time'
df["start_time"] = (
    df["start_time"]
    .str.replace('"', "")
    .str.replace("Start Time:", "")
    .str.replace(" p.m. Local", "pm")
    .str.replace(" a.m. Local", "am")
)

week_day, date, year = df["date"].str.split(", ").str
df["week_day"] = week_day
df["date"] = date
df["year"] = year

game_type, fiedl_type = df["game_type"].str.split(", ").str
df["game_type"] = game_type
df["fiedl_type"] = fiedl_type

print(df)

# profile = df.profile_report(title="Test")
# profile.to_file(output_file="profiler.html")

df.to_csv("New1.csv", index=False)

# Aplicar un labelEncoder
df = df.apply(LabelEncoder().fit_transform)

# Adding a new feature 'attendance_squared'
df["attendance_squared"] = df["attendance"] ** 2

# Importing the dataset
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Ajustar el modelo de regresión lineal a los datos que tenemos
reg = LinearRegression().fit(X, y)

# Calculo del parametro R2
y_pred = reg.predict(X)
r2 = r2_score(y, y_pred)
print("R2 score:", r2)

# Getting the coefficients
coeff = reg.coef_

# Adding the constant term to the coefficients
coeff = np.insert(coeff, 0, reg.intercept_)

# Creating the equation
equation = "y = "
for i, c in enumerate(coeff):
    if i == 0:
        equation += f"{c}"
    elif c >= 0:
        equation += f" + {c} * x_{i}"
    else:
        equation += f" - {abs(c)} * x_{i}"

print("Coefficients:", coeff)
print("Equation:", equation)
