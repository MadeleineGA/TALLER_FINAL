import requests
import pandas as pd
import mlflow
import mlflow.sklearn
import yaml

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from mlflow.models import infer_signature


# -----------------------------
# Cargar configuración
# -----------------------------
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

url = config["data"]["api_url"]

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])

# -----------------------------
# Cargar datos desde API
# -----------------------------
response = requests.get(url)
response.raise_for_status()

data = response.json()
records = data["result"]["records"]

df = pd.DataFrame(records)

print("Datos cargados:")
print(df.head())
print(df.shape)

# -----------------------------
# Limpieza básica
# -----------------------------
df = df.drop(columns=["entry_id"], errors="ignore")

df["trafico"] = pd.to_numeric(df["trafico"], errors="coerce")
df["anno"] = pd.to_numeric(df["anno"], errors="coerce")
df["trimestre"] = pd.to_numeric(df["trimestre"], errors="coerce")
df["mes_del_trimestre"] = pd.to_numeric(df["mes_del_trimestre"], errors="coerce")

# eliminar los reportes de tráfico en cero
if config["data"]["dropna"]:
    df = df.dropna(subset=[config["data"]["target"]])

# Quitar registros con tráfico negativo si existieran - en caso de que se registre alguno a futuro por error
df = df[df["trafico"] >= 0]

# -----------------------------
# Variables predictoras y objetivo
# -----------------------------
X = df.drop(columns=["trafico"])
y = df["trafico"]

categorical_features = ["id_empresa", "empresa"]
numeric_features = ["anno", "trimestre", "mes_del_trimestre"]

# -----------------------------
# Preprocesamiento
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numeric_features),
    ]
)

# -----------------------------
# Modelo
# -----------------------------
model = RandomForestRegressor(
    n_estimators=config["model"]["n_estimators"],
    random_state=config["model"]["random_state"]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# -----------------------------
# Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config["model"]["test_size"],
    random_state=config["model"]["random_state"]
)

# -----------------------------
# Entrenamiento
# -----------------------------
pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R2: {r2}")

# -----------------------------
# MLflow tracking
# -----------------------------
signature = infer_signature(X_test, preds)
input_example = X_test.head(3)

with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", config["model"]["n_estimators"])
    mlflow.log_param("test_size", config["model"]["test_size"])

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

print("✅ Entrenamiento finalizado y modelo registrado en MLflow.")