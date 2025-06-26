from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import numpy as np

app = Flask(__name__)

# 1. Cargar el modelo
model = xgb.XGBRegressor()
model.load_model("modelo_guardado.json")

# 2. Columnas usadas en el modelo
model_columns = [
    'año', 'mes', 'Abancay', 'Arequipa', 'Ayacucho', 'Cajamarca', 'Callao',
    'Cerro de Pasco', 'Chachapoyas', 'Chiclayo', 'Cusco', 'Huancavelica',
    'Huancayo', 'Huaraz', 'Huánuco', 'Ica', 'Iquitos', 'Lima', 'Moquegua',
    'Moyobamba', 'Piura', 'Pucallpa', 'Puerto Maldonado', 'Puno', 'Tacna',
    'Trujillo', 'Tumbes', 'mes_sin', 'mes_cos', 'Invierno', 'Otoño',
    'Primavera', 'Verano'
]

# 3. Lista de ciudades
ciudades = [
    'Abancay', 'Arequipa', 'Ayacucho', 'Cajamarca', 'Callao', 'Cerro de Pasco',
    'Chachapoyas', 'Chiclayo', 'Cusco', 'Huancavelica', 'Huancayo', 'Huaraz',
    'Huánuco', 'Ica', 'Iquitos', 'Lima', 'Moquegua', 'Moyobamba', 'Piura',
    'Pucallpa', 'Puerto Maldonado', 'Puno', 'Tacna', 'Trujillo', 'Tumbes'
]

# 4. Ruta para predicción simplificada
@app.route("/predecir", methods=["POST"])
def predecir():
    try:
        data = request.get_json()

        anio = int(data.get("año"))
        mes = int(data.get("mes"))
        estacion = data.get("estacion")
        ciudad = data.get("ciudad")

        if estacion not in ["Invierno", "Verano", "Primavera", "Otoño"]:
            return jsonify({"error": "Estación no válida"}), 400
        if ciudad not in ciudades:
            return jsonify({"error": "Ciudad no válida"}), 400

        # 5. Crear input_data igual que el script original
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

        # 6. Convertir a DataFrame y reordenar
        input_df = pd.DataFrame(input_data)[model_columns]

        # 7. Predecir
        pred = model.predict(input_df)[0]
        temp = float(round(pred, 2))

        return jsonify({
            "año": anio,
            "mes": mes,
            "ciudad": ciudad,
            "temperatura_maxima_predicha": temp
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
