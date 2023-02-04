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

# Create dummies for the categorical variables
df = pd.get_dummies(
    df,
    columns=[
        "attendance",
        "away_team",
        "away_team_errors",
        "away_team_hits",
        "away_team_runs",
        "date",
        "game_duration",
        "game_type",
        "home_team",
        "home_team_errors",
        "home_team_hits",
        "home_team_runs",
        "start_time",
        "venue",
        "week_day",
        "year",
        "fiedl_type",
    ],
)

# Save the dataframe
df.to_csv("dummy_dataset.csv", index=False)


# Mostrat la ecuacion
reg = LinearRegression().fit(X, y)
coefficients = reg.coef_

interception = reg.intercept_
equation = "y = "
for i in range(len(coefficients)):
    equation += str(coefficients[i]) + " * X" + str(i + 1) + " + "
equation += str(interception)
print(equation)

# Data to predict
data_to_predict = pd.DataFrame(
    {
        "home_team": ["New York Mets"],
        "away_team": ["San Francisco Giants"],
        "week_day": ["Sunday"],
        "start_time": ["7:38pm"],
        "game_type": ["Night Game"],
        "venue": ["Kauffman Stadium"],
        "year": [2016],
        "col7": [0],
        "col8": [0],
        "col9": [0],
        "col10": [0],
        "col11": [0],
        "col12": [0],
        "col13": [0],
        "col14": [0],
        "col15": [0],
        "col16": [0],
    }
)

# Apply LabelEncoder to the data to predict
data_to_predict = data_to_predict.apply(LabelEncoder().fit_transform)

# Adding the 'attendance_squared' feature
data_to_predict["attendance_squared"] = 0

# Converting the input data into a numpy array
X_to_predict = data_to_predict.iloc[:, :-1].values

# Using the model to make the prediction
attendance = reg.predict(X_to_predict)

# Displaying the predicted attendance
print("The predicted attendance for the game is:", int(attendance[0]))
