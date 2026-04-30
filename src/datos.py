# Leer la base de datos para el entrenamiento del modelo
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://www.postdata.gov.co/api/action/datastore/search.json?resource_id=d40c5e75-db56-4ec1-a441-0314c47bd71d&limit=1100"

response = requests.get(url)
data = response.json()

# Extraer solo los registros
records = data["result"]["records"]

# Convertir a DataFrame
df = pd.DataFrame(records)

# 🔧 LIMPIEZA DE DATOS
df = df.drop(columns=["entry_id", "mes_grap"], errors="ignore")

# Convertir tráfico a número
df["trafico"] = pd.to_numeric(df["trafico"], errors="coerce")


# 🔎 VERIFICACIÓN FINAL
print(df.head())
print(df.columns)
print(df.shape)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# -----------------------------
# 🔎 INFO GENERAL
# -----------------------------
print("\n--- INFO ---")
print(df.info())

print("\n--- DESCRIBE GENERAL ---")
print(df.describe())

# -----------------------------
# 🔎 NULOS
# -----------------------------
print("\n--- NULOS ---")
print(df.isnull().sum())

# -----------------------------
# 📊 DESCRIBE POR EMPRESA
# -----------------------------
print("\n--- DESCRIBE POR EMPRESA ---")
print(df.groupby("empresa")["trafico"].describe())

# -----------------------------
# 📊 DISTRIBUCIÓN DEL TRÁFICO
# -----------------------------
plt.figure(figsize=(10,5))
sns.histplot(df["trafico"], bins=30)
plt.title("Distribución del tráfico")
plt.xlabel("Tráfico")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("outputs/trafico_distribucion.png")
plt.show()

# -----------------------------
# 📊 CONTEO POR EMPRESA
# -----------------------------
print("\n--- CONTEO POR EMPRESA ---")
print(df["empresa"].value_counts())

# -----------------------------
# 📊 PROMEDIO DE TRÁFICO POR EMPRESA
# -----------------------------
traf_pro = df.groupby("empresa")["trafico"]\
    .mean()\
    .sort_values(ascending=False)

plt.figure(figsize=(12,6))
traf_pro.head(10).plot(kind="bar")

plt.title("Promedio de tráfico por empresa")
plt.ylabel("Tráfico promedio")
plt.xlabel("Empresa")
plt.xticks(rotation=45, fontsize=8)

plt.tight_layout()
plt.savefig("outputs/trafico_empresa.png")
plt.show()

# -----------------------------
# 📊 TABLA POR AÑO Y TRIMESTRE
# -----------------------------
tabla_tiempo = df.groupby(["anno", "trimestre"])["trafico"]\
    .sum()\
    .reset_index()

print("\n--- TRÁFICO POR AÑO Y TRIMESTRE ---")
print(tabla_tiempo.head(20))

# -----------------------------
# 📊 GRÁFICA TEMPORAL
# -----------------------------
tabla_tiempo["periodo"] = tabla_tiempo["anno"].astype(str) + "-T" + tabla_tiempo["trimestre"].astype(str)

plt.figure(figsize=(14,8))

sns.lineplot(
    data=tabla_tiempo,
    x="periodo",
    y="trafico"
)

plt.xticks(rotation=45)
plt.title("Evolución del tráfico en el tiempo")
plt.xlabel("Periodo (Año-Trimestre)")
plt.ylabel("Tráfico")

plt.tight_layout()
plt.savefig("outputs/trafico_tiempo.png")
plt.show()
# -----------------------------
# ELIMINAR VALORES NULOS
# -----------------------------
df = df.dropna(subset=["trafico"])

print("\n--- INFO ---")
print(df.info())
