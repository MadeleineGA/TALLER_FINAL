Taller final MLOPs

Universidad EAN – Maestría en Ciencia de Datos

Estudiante: Madeleine Gil

Este proyecto desarrolla un pipeline de machine learning para predecir el tráfico de internet móvil en Colombia a partir de datos abiertos obtenidos desde una API pública.

El flujo integra prácticas de MLOps, el cual incluye:

Ingesta de datos desde API
Preprocesamiento
Entrenamiento de modelo
Evaluación
Tracking con MLflow
Automatización mediante CI/CD con GitHub Actions

La estructura del proyecto:
<img width="311" height="473" alt="image" src="https://github.com/user-attachments/assets/e12e9393-ea80-48e7-922a-2f6e0cef3685" />

Dataset:

Los datos se obtienen desde una API pública del portal de datos abiertos postdata: https://www.postdata.gov.co/resource/trafico-de-internet-movil-de-cargo-fijo

Tipo: Tráfico de internet móvil por cargo fijo
Variables principales:
- anno
- trimestre
- mes_del_trimestre
- id_empresa
- empresa
- trafico

Configuración:

El archivo config.yaml define:

- URL de la API
- Variable objetivo
- Parámetros del modelo
- Configuración de MLflow

Pipeline de Machine Learning:

1. Preprocesamiento:
- Conversión de variables a tipo numérico
- Eliminación de valores nulos
- Codificación de variables categóricas (OneHotEncoding)
- Escalamiento de variables numéricas

2. Modelo:
Se utilizó un modelo de Random Forest Regressor, adecuado para capturar relaciones no lineales y manejar datos heterogéneos.

3. Evaluación:
Se calcularon las siguientes métricas:

- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coeficiente de determinación)

Tracking con MLflow

Se utiliza MLflow para:

- Registrar parámetros
- Guardar métricas
- Almacenar el modelo entrenado
- Registrar firma del modelo
