from flask import Flask, request, render_template
import pickle
import numpy as np
from regresionLineal import entrenar_modelo

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/formulario')
def formulario():
    return render_template('index.html')

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

        entrada = np.array([[año, mes, urbano, tipo_punto]])
        prediccion = modelo.predict(entrada)
        resultado = int(round(prediccion[0]))

        return render_template('index.html', resultado=f"Predicción: {resultado} animales por vacunar.")
    except Exception as e:
        return render_template('index.html', resultado=f"Error en la predicción: {e}")
