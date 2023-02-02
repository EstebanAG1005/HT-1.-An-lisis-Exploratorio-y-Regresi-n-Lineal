import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#1 Haga una exploración rápida de sus datos para eso haga un resumen de su dataset.

# Importar el dataset
df = pd.read_csv("baseball.csv")

# Ver las primeras filas
print(df.head())

# Ver las estadísticas básicas
print(df.describe())

# Ver los tipos de datos de las columnas
print(df.dtypes)

# Verificar valores faltantes
print(df.isna().sum())

#1.2 Diga el tipo de cada una de las variables del dataset (cualitativa o categórica, cuantitativa continua, cuantitativa discreta)

# Clasificar cada columna según su tipo de datos
for col in df.columns:
    if df[col].dtype == "object":
        print("La columna '{}' es cualitativa o categórica.".format(col))
    elif df[col].dtype == "float64" or df[col].dtype == "int64":
        if len(df[col].unique()) > 10:
            print("La columna '{}' es cuantitativa continua.".format(col))
        else:
            print("La columna '{}' es cuantitativa discreta.".format(col))

#1.3 Incluya los gráficos exploratorios siendo consecuentes con el tipo de variable que están representando.

# Iterar sobre cada columna del dataset
for column in df.columns:
    # Determinar el tipo de variable de la columna
    dtype = df[column].dtype
    
    if dtype == "object":
        # Categórica: hacer un gráfico de barras o pastel
        frequencies = df[column].value_counts()
        frequencies.plot(kind="bar")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.title("Frequency of " + column)
        plt.show()
    elif dtype == "float" or dtype == "int":
        # Cuantitativa: hacer un histograma, boxplot o scatter plot
        plt.hist(df[column])
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.title("Histogram of " + column)
        plt.show()
    else:
        # Otro tipo de variable
        print("Column " + column + " has an unknown data type: " + str(dtype))

#1.4 Aísle las variables numéricas de las categóricas, haga un análisis de correlación entre las mismas.

# Aislar las variables numéricas
numeric_vars = df.select_dtypes(include=["float", "int"])

# Hacer un análisis de correlación entre las variables numéricas
corr = numeric_vars.corr()

# Mostrar la matriz de correlación
print(corr)

# Hacer un gráfico de calor de la matriz de correlación
plt.imshow(corr, cmap="RdBu", vmin=-1, vmax=1)
plt.colorbar()
plt.show()

# Encontrar todas las variables categóricas
cat_vars = df.select_dtypes(include=["object"]).columns

#1.5 Utilice las variables categóricas, haga tablas de frecuencia, proporción, gráficas de barras o cualquier otra técnica que le permita explorar los datos.

# Recorrer todas las variables categóricas
for var in cat_vars:
    # Hacer una tabla de frecuencia de la variable categórica
    freq = df[var].value_counts()

    # Mostrar la tabla de frecuencia
    print(freq)

    # Hacer una tabla de proporción de la variable categórica
    prop = df[var].value_counts(normalize=True)

    # Mostrar la tabla de proporción
    print(prop)

    # Hacer un gráfico de barra de la tabla de frecuencia
    plt.bar(freq.index, freq.values)
    plt.xlabel("Valores de la variable")
    plt.ylabel("Frecuencia")
    plt.title(f"Gráfico de barra de frecuencia de la variable categórica {var}")
    plt.show()

    # Hacer un gráfico de barra de la tabla de proporción
    plt.bar(prop.index, prop.values)
    plt.xlabel("Valores de la variable")
    plt.ylabel("Proporción")
    plt.title(f"Gráfico de barra de proporción de la variable categórica {var}")
    plt.show()

#1.6 Realice la limpieza de variables utilizando las técnicas vistas en clase, u otras que piense pueden ser de utilidad

# Reemplazar los valores nulos con un 0
df = df.fillna(0)

# Seleccionar solo las columnas numéricas
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Iterar sobre las columnas numéricas
for col in numeric_cols:
    # Calcular el promedio y la desviación estándar de la columna
    mean = df[col].mean()
    std = df[col].std()

    # Truncar valores atípicos a 2 desviaciones estándar
    df[col] = np.clip(df[col], mean - 2*std, mean + 2*std)
