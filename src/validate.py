import mlflow
import mlflow.pyfunc
import pandas as pd
import requests
import yaml
import sys

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# -----------------------------
# Config
# -----------------------------
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

url = config["data"]["api_url"]
target = config["data"]["target"]

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])

# -----------------------------
# Cargar datos desde API
# -----------------------------
print("🔎 Cargando datos...")

response = requests.get(url)
data = response.json()

records = data["result"]["records"]
df = pd.DataFrame(records)

# -----------------------------
# Limpieza (igual que train)
# -----------------------------
df = df.drop(columns=["entry_id"], errors="ignore")

df["trafico"] = pd.to_numeric(df["trafico"], errors="coerce")
df["anno"] = pd.to_numeric(df["anno"], errors="coerce")
df["trimestre"] = pd.to_numeric(df["trimestre"], errors="coerce")
df["mes_del_trimestre"] = pd.to_numeric(df["mes_del_trimestre"], errors="coerce")

df = df.dropna(subset=["trafico"])

# -----------------------------
# Variables
# -----------------------------
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config["model"]["test_size"],
    random_state=config["model"]["random_state"]
)


# -----------------------------
# Cargar último modelo de MLflow
# -----------------------------
print("🔎 Cargando modelo desde MLflow...")

runs = mlflow.search_runs(order_by=["start_time DESC"])

if runs.empty:
    print("❌ No hay modelos registrados")
    sys.exit(1)

run_id = runs.iloc[0]["run_id"]
model_uri = f"runs:/{run_id}/model"

print(f"Modelo: {model_uri}")

model = mlflow.pyfunc.load_model(model_uri)


# -----------------------------
# Predicción
# -----------------------------
print("🔎 Evaluando modelo...")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R2: {r2}")


# -----------------------------
# Validación
# -----------------------------
THRESHOLD = 0.5  # ajusta según tu modelo

if r2 >= THRESHOLD:
    print("✅ Modelo aprobado")
    sys.exit(0)
else:
    print("❌ Modelo no cumple el umbral")
    sys.exit(1)