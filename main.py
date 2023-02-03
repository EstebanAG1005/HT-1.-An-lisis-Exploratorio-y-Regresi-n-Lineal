import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def display_menu():
    print("Menu:")
    print("1. Haga una exploración rápida de sus datos para eso haga un resumen de su dataset.")
    print("2. Diga el tipo de cada una de las variables del dataset (cualitativa o categórica, cuantitativa continua, cuantitativa discreta)")
    print("3. Incluya los gráficos exploratorios siendo consecuentes con el tipo de variable que están representando.")
    print("4. Aísle las variables numéricas de las categóricas, haga un análisis de correlación entre las mismas.")
    print("5. Utilice las variables categóricas, haga tablas de frecuencia, proporción, gráficas de barras o cualquier otra técnica que le permita explorar los datos.")
    print("6. Realice la limpieza de variables utilizando las técnicas vistas en clase, u otras que piense pueden ser de utilidad")
    print("7. Salir")

def main():
    while True:
        display_menu()
        opcion = int(input("Elija una opción: "))
        
        if opcion == 1:

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
        
        elif opcion == 2:
            # Importar el dataset
            df = pd.read_csv("baseball.csv")

            # Clasificar cada columna según su tipo de datos
            for col in df.columns:
                if df[col].dtype == "object":
                    print("La columna '{}' es cualitativa o categórica.".format(col))
                elif df[col].dtype == "float64" or df[col].dtype == "int64":
                    if len(df[col].unique()) > 10:
                        print("La columna '{}' es cuantitativa continua.".format(col))
                    else:
                        print("La columna '{}' es cuantitativa discreta.".format(col))

        elif opcion == 3:

            # Importar el dataset
            df = pd.read_csv("baseball.csv")

            # Iterar sobre cada columna del dataset
            for column in df.columns:
                # Determinar el tipo de variable de la columna
                dtype = df[column].dtype

                if dtype == "object":
                    # Categórica: hacer un gráfico de barras o pastel
                    frequencies = df[column].value_counts()
                    if len(frequencies) > 15:
                        frequencies = frequencies[:15]
                    frequencies.plot(kind="bar")
                    plt.xticks(rotation=90)
                    plt.xlabel(column)
                    plt.ylabel("Frequency")
                    plt.title("Frequency of " + column)
                    plt.show()
                elif dtype == "float" or dtype == "int":
                    # Cuantitativa: hacer un histograma, boxplot o scatter plot
                    df[column].fillna(value=0, inplace=True)
                    plt.hist(df[column])
                    plt.xlabel(column)
                    plt.ylabel("Frequency")
                    plt.title("Histogram of " + column)
                    plt.show()
                else:
                    # Otro tipo de variable
                    print("Column " + column + " has an unknown data type: " + str(dtype))


        elif opcion == 4:

            # Importar el dataset
            df = pd.read_csv("baseball.csv")

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

        elif opcion == 5:

            # Importar el dataset
            df = pd.read_csv("baseball.csv")

            # Encontrar todas las variables categóricas
            cat_vars = df.select_dtypes(include=["object"]).columns
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

                # Verificar si hay más de 15 valores en la variable categórica
                if len(freq) > 15:
                    # Si hay más de 15 valores, mostrar solo los primeros 15 valores
                    freq = freq[:15]
                    prop = prop[:15]

                # Hacer un gráfico de barra de la tabla de frecuencia
                plt.bar(freq.index, freq.values)
                plt.xlabel("Valores de la variable")
                plt.ylabel("Frecuencia")
                plt.title(f"Gráfico de barra de frecuencia de la variable categórica {var}")
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.show()

                # Hacer un gráfico de barra de la tabla de proporción
                plt.bar(prop.index, prop.values)
                plt.xlabel("Valores de la variable")
                plt.ylabel("Proporción")
                plt.title(f"Gráfico de barra de proporción de la variable categórica {var}")
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.show()

            
        elif opcion == 6:
            # Se carga la data
            df = pd.read_csv("baseball.csv")

            # Remove comma from 'attendance' column
            df["attendance"] = (
                df["attendance"]
                .str.replace(",", "", regex=False)
                .str.replace("'", "", regex=False)
                .str.replace("]", "", regex=False)
            )


            df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")
            df = df.dropna(subset=["attendance"])

            # Remove quotes from 'other_info_string' column
            df = df.drop("other_info_string", axis=1)

            # Remove quotes from 'other_info_string' column
            df["venue"] = df["venue"].str.replace(":", "")

            df["game_duration"] = df["game_duration"].str.replace(":", "")

            # Remove quotes from 'start_time' column
            df["start_time"] = df["start_time"].str.replace('"', "")


            print(df)

            # Guarda los cambios en un archivo csv
            df.to_csv("New.csv", index=False)


        elif opcion == 7:
            break
        else:
            print("Opción inválida.")

if __name__ == "__main__":
    main()

