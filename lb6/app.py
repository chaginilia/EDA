from flask import Flask, request, jsonify
import pandas as pd
from catboost import CatBoostRegressor
from eda import transform_data
import os

app = Flask(__name__)

model = None

def load_model():
    global model
    if model is None:
        model_path = "/app/trained_model/trained_model.cbm"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        model = CatBoostRegressor()
        model.load_model(model_path)
        print("Модель успешно загружена")
    return model

@app.route("/predict", methods=["POST"])
def predict():
    current_model = load_model()
    
    content = request.json
    df = pd.DataFrame(content, index=[0])
    df = transform_data(df)
    
    result = current_model.predict(df)[0]
    return jsonify({"prediction_Rings": float(result)})

@app.route("/health")
def health():
    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)