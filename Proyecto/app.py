from flask import Flask, request, render_template
import pickle
import pandas as pd
import json
from regresionLineal import entrenar_modelo, evaluar_modelo
from conexionRenderBd import get_render_connection

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/documentacion")
def documentacion():
    return render_template("documentacion.html")

@app.route("/comprension_datos")
def comprension():
    return render_template("comprension_datos.html")

@app.route("/ingenieria_datos")
def ingenieriaDatos():
    return render_template("ingenieria_datos.html")

@app.route("/ingenieria_modelo")
def ingenieriaModelo():
    return render_template("ingenieria_modelo.html")

@app.route("/implementacion")
def implementacion():
    mse, r2 = evaluar_modelo()
    return render_template('implementacion.html', mse=round(mse, 2), r2=round(r2, 2))

@app.route('/entrenar')
def entrenar():
    entrenar_modelo()
    return render_template('entrenado.html')

@app.route('/entrenardo', methods=['GET', 'POST'])
def subir_datos():
    mensaje = ''
    if request.method == 'POST':
        archivo = request.files['archivo']
        if archivo:
            try:
                if archivo.filename.endswith('.csv'):
                    df = pd.read_csv(archivo)
                elif archivo.filename.endswith('.xlsx'):
                    df = pd.read_excel(archivo)
                else:
                    mensaje = 'Formato no soportado'
                    return render_template('subir_datos.html', mensaje=mensaje)

                conn = get_render_connection()
                df.to_sql('vacunaciones', conn, if_exists='append', index=False)
                conn.close()
                mensaje = 'Datos cargados exitosamente.'
            except Exception as e:
                mensaje = f'Error al procesar el archivo: {e}'

    return render_template('subir_datos.html', mensaje=mensaje)


@app.route('/formulario')
def formulario():
    return render_template('prediccion.html')



@app.route('/predecir', methods=['POST'])
def predecir():
    entrenar_modelo()
    with open('modelo.pkl', 'rb') as f:
        modelo = pickle.load(f)
    try:
        año = int(request.form['año'])
        mes = int(request.form['mes'])
        urbano = int(request.form['urbano'])
        tipo_punto = int(request.form['tipo_punto'])
        sector = int(request.form['sector'])

        entrada_df = pd.DataFrame([[año, mes, urbano, tipo_punto, sector]],
            columns=['Año', 'Mes', 'Urbano o Rural?', 'Es vivienda o punto fijo?', 'Sector codificado']
        )

        prediccion = modelo.predict(entrada_df)
        resultado = int(round(prediccion[0]))

        entrada = {
            "Año": año,
            "Mes": mes,
            "Urbano o Rural?": urbano,
            "Es vivienda o punto fijo?": tipo_punto,
            "Sector codificado": sector
        }
        entrada_json = json.dumps(entrada)
        guardar_prediccion(entrada_json, resultado)

        return render_template('prediccion.html', resultado=f"Predicción: {resultado} animales por vacunar.")
    except Exception as e:
        return render_template('prediccion.html', resultado=f"Error en la predicción: {e}")
    
def guardar_prediccion(entrada_json, resultado):
    conn = get_render_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predicciones (entrada_json, resultado_predicho)
        VALUES (%s, %s)
    """, (entrada_json, resultado))
    conn.commit()
    conn.close()

@app.route('/historial')
def historial():
    conn = get_render_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predicciones ORDER BY fecha DESC")
    filas = cursor.fetchall()
    conn.close()

    predicciones = []
    for fila in filas:
        entrada_dict = json.loads(fila[1])
        resultado = fila[2]
        fecha = fila[3]
        predicciones.append({
            'entrada': entrada_dict,
            'resultado': resultado,
            'fecha': fecha
        })

    return render_template('predicciones.html', predicciones=predicciones)
