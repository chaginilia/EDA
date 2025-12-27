import os
from catboost import CatBoostRegressor
from explainerdashboard import RegressionExplainer
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from eda import prepare_training_data
import joblib

MODEL_DIR = "/app/trained_model"
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.cbm")
EXPLAINER_FILE = os.path.join(MODEL_DIR, "explainer.joblib")

while not os.path.exists(MODEL_PATH):
    print("Ждём обученную модель...")
    import time
    time.sleep(3)

if os.path.exists(EXPLAINER_FILE):
    print("Explainer уже сгенерирован — пропускаем")
    exit(0)

print("Загружаем данные и модель...")
abalone = fetch_ucirepo(id=1)
df = abalone.data.features.copy()
df["Rings"] = abalone.data.targets

df_prepared = prepare_training_data(df)
X = df_prepared.drop("Rings", axis=1)
y = df_prepared["Rings"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostRegressor()
model.load_model(MODEL_PATH)

print("Создаём Explainer...")

explainer = RegressionExplainer(
    model,
    X_test,
    y_test,
    shap="guess"
)

print(f"Сохраняем explainer → {EXPLAINER_FILE}")
joblib.dump(explainer, EXPLAINER_FILE)

print("Explainer успешно сохранён!")
