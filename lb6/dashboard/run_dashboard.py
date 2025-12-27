import os
import time
import joblib
from explainerdashboard import ExplainerDashboard

EXPLAINER_FILE = "/app/trained_model/explainer.joblib"

print("Ждём explainer.joblib...")
while not os.path.exists(EXPLAINER_FILE):
    time.sleep(2)
    print(".", end="", flush=True)

print("\nДашборд на http://0.0.0.0:8050")

explainer = joblib.load(EXPLAINER_FILE)

dashboard = ExplainerDashboard(
    explainer,
    title="Abalone Age Prediction",
    name="Abalone",
    model_summary="CatBoost Regressor",
    shap_dependence=True,
    shap_interaction=False,
)

dashboard.run(host="0.0.0.0", port=8050, use_waitress=True)