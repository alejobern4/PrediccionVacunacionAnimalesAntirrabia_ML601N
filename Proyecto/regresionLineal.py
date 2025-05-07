import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import os

def entrenar_modelo():

    # Obtener la ruta absoluta al archivo CSV
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    ruta_archivo = os.path.join(ruta_base, "Datos", "Censo_de_vacunaci_n_antirr_bia__alcald_a_de_ch_a_20250430.csv")

    # Leer el archivo
    df = pd.read_csv(ruta_archivo)

    # Eliminar filas con datos faltantes
    df = df.dropna()

    # Convertir nombres de meses a números
    meses = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
        'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
        'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    df['Mes'] = df['Mes'].str.lower().map(meses)

    # Codificar variables categóricas
    df['Urbano o Rural?'] = df['Urbano o Rural?'].map({'Rural': 0, 'Urbano': 1})
    df['Es vivienda o punto fijo?'] = df['Es vivienda o punto fijo?'].map({'Vivienda': 0, 'Punto fijo': 1})

    # Crear columna objetivo (y): total vacunados
    df['Vacunados totales'] = df['Vacunados (dogs)'] + df['Vacunados (cats)']

   # Elimina filas con valores faltantes en las columnas relevantes
    df = df.dropna(subset=['Año', 'Mes', 'Urbano o Rural?', 'Es vivienda o punto fijo?', 'Perros totales', 'Gatos totales', 'Vacunados totales'])

    # Define las variables independientes
    X = df[['Año', 'Mes', 'Urbano o Rural?', 'Es vivienda o punto fijo?']]
    y = df['Vacunados totales']



    # Separar en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar modelo de regresión lineal
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Guardar el modelo entrenado
    with open('modelo.pkl', 'wb') as f:
        pickle.dump(modelo, f)