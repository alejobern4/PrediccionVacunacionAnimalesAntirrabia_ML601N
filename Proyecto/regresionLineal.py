import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

#Ruta al dataset
ruta_base = os.path.dirname(os.path.abspath(__file__))
ruta_archivo = os.path.join(ruta_base, "Datos", "Censo_de_vacunaci_n_antirr_bia__alcald_a_de_ch_a_20250430.csv")

def entrenar_modelo():
    #Leer el archivo csv
    df = pd.read_csv(ruta_archivo)

    #Normalizar y codificar sector/vereda
    df['Sector /Vereda'] = df['Sector /Vereda'].str.lower().str.strip()

    sector_map = {
        'cerca de pedra': 0,
        'centro': 1,
        '16_centro': 1,
        'la balsa': 2,
        'tiquiza': 3,
        'tiqueza': 3,
        'fonqueta': 4,
        'fusca': 5,
        'fagua': 6,
        'samaria': 7,
        'otro': 8
    }

    df['Sector codificado'] = df['Sector /Vereda'].map(sector_map)

    df = df.dropna()
    meses = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
        'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
        'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    df['Mes'] = df['Mes'].str.lower().map(meses)
    df['Urbano o Rural?'] = df['Urbano o Rural?'].map({'Rural': 0, 'Urbano': 1})
    df['Es vivienda o punto fijo?'] = df['Es vivienda o punto fijo?'].map({'Vivienda': 0, 'Punto fijo': 1})
    df['Vacunados totales'] = df['Vacunados (dogs)'] + df['Vacunados (cats)']

    # Asegurarse de que no haya valores nulos en las variables necesarias
    df = df.dropna(subset=['Año', 'Mes', 'Urbano o Rural?', 'Es vivienda o punto fijo?', 'Vacunados totales', 'Sector codificado'])

    # Variables predictoras y objetivo
    X = df[["Año", "Mes", "Urbano o Rural?", "Es vivienda o punto fijo?", "Sector codificado"]]
    y = df['Vacunados totales']




    #Division del dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Entrenar modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    #Guardar modelo entrenado
    with open('modelo.pkl', 'wb') as f:
        pickle.dump(modelo, f)

    #Tambien guardamos los datos de test para la evaluacion
    ruta_basedir = os.path.dirname(os.path.abspath(__file__))

    carpeta_destino = os.path.join(ruta_basedir, 'Datos')

    os.makedirs(carpeta_destino, exist_ok=True)

    X_test.to_csv(os.path.join(carpeta_destino, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(carpeta_destino, 'y_test.csv'), index=False)

def evaluar_modelo():
    #Cargar modelo
    with open('modelo.pkl', 'rb') as f:
        modelo = pickle.load(f)

    #Cargar datos de prueba
    ruta_basex = os.path.dirname(os.path.abspath(__file__))
    ruta_archivox = os.path.join(ruta_basex, "Datos", "X_test.csv")
    X_test = pd.read_csv(ruta_archivox)

    ruta_basey = os.path.dirname(os.path.abspath(__file__))
    ruta_archivoy = os.path.join(ruta_basey, "Datos", "y_test.csv")
    y_test = pd.read_csv(ruta_archivoy)

    #Prediccion y evaluacion
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2
