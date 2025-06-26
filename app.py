from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# --- Ruta de inicio para verificar que la API está viva ---
@app.route("/", methods=["GET"])
def home():
    return "✅ API Clima funcionando correctamente en Heroku."

# --- Intentar cargar el modelo ---
try:
    model = xgb.XGBRegressor()
    model.load_model("modelo_guardado.json")
except Exception as e:
    model = None
    print(f"❌ Error al cargar el modelo: {e}")

# --- Columnas del modelo ---
model_columns = [
    'año', 'mes', 'Abancay', 'Arequipa', 'Ayacucho', 'Cajamarca', 'Callao',
    'Cerro de Pasco', 'Chachapoyas', 'Chiclayo', 'Cusco', 'Huancavelica',
    'Huancayo', 'Huaraz', 'Huánuco', 'Ica', 'Iquitos', 'Lima', 'Moquegua',
    'Moyobamba', 'Piura', 'Pucallpa', 'Puerto Maldonado', 'Puno', 'Tacna',
    'Trujillo', 'Tumbes', 'mes_sin', 'mes_cos', 'Invierno', 'Otoño',
    'Primavera', 'Verano'
]

# --- Lista de ciudades disponibles ---
ciudades = [
    'Abancay', 'Arequipa', 'Ayacucho', 'Cajamarca', 'Callao', 'Cerro de Pasco',
    'Chachapoyas', 'Chiclayo', 'Cusco', 'Huancavelica', 'Huancayo', 'Huaraz',
    'Huánuco', 'Ica', 'Iquitos', 'Lima', 'Moquegua', 'Moyobamba', 'Piura',
    'Pucallpa', 'Puerto Maldonado', 'Puno', 'Tacna', 'Trujillo', 'Tumbes'
]

# --- Estación automática según mes ---
def obtener_estacion(mes):
    if mes in [12, 1, 2]:
        return "Verano"
    elif mes in [3, 4, 5]:
        return "Otoño"
    elif mes in [6, 7, 8]:
        return "Invierno"
    elif mes in [9, 10, 11]:
        return "Primavera"
    return None

# --- Ruta de predicción ---
@app.route("/predecir", methods=["POST"])
def predecir():
    if model is None:
        return jsonify({"error": "Modelo no cargado correctamente."}), 500
    try:
        data = request.get_json()
        anio = int(data.get("año"))
        mes = int(data.get("mes"))
        ciudad = data.get("ciudad")

        if ciudad not in ciudades:
            return jsonify({"error": "Ciudad no válida"}), 400
        if mes < 1 or mes > 12:
            return jsonify({"error": "Mes fuera de rango (1-12)"}), 400

        estacion = obtener_estacion(mes)

        input_data = {
            'año': [anio],
            'mes': [mes],
            'mes_sin': [np.sin(2 * np.pi * mes / 12)],
            'mes_cos': [np.cos(2 * np.pi * mes / 12)],
            'Invierno': [1 if estacion == 'Invierno' else 0],
            'Otoño': [1 if estacion == 'Otoño' else 0],
            'Primavera': [1 if estacion == 'Primavera' else 0],
            'Verano': [1 if estacion == 'Verano' else 0]
        }

        for c in ciudades:
            input_data[c] = [1 if c == ciudad else 0]

        input_df = pd.DataFrame(input_data)[model_columns]
        pred = model.predict(input_df)[0]
        temp = float(round(pred, 2))

        return jsonify({
            "año": anio,
            "mes": mes,
            "ciudad": ciudad,
            "estacion_detectada": estacion,
            "temperatura_maxima_predicha": temp
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Para desarrollo local, ignorado por gunicorn ---
if __name__ == "__main__":
    app.run(debug=True)
