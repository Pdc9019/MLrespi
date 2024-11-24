import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

# Rutas de los archivos
ruta_emergencias = r"C:\Users\benja\Documents\GitHub\MLrespi\Datos\Datos_urgencias_respiratorias_reducido.csv"

ruta_calidad_aire = r"C:\Users\benja\Documents\GitHub\MLrespi\Datos\MP25\elbosque_semanal.csv"


# Cargar los datos
data_emergencias = pd.read_csv(ruta_emergencias)
data_calidad_aire = pd.read_csv(ruta_calidad_aire)

# Filtrar para eliminar años 2020 y 2021 (por el covid 19)
data_emergencias = data_emergencias[~data_emergencias['Anio'].isin([2020, 2021])]
data_calidad_aire = data_calidad_aire[~data_calidad_aire['Anio'].isin([2020, 2021])]

# Combinar los datasets por año y semana
data_combinada = pd.merge(data_emergencias, data_calidad_aire, on=['Anio', 'SemanaEstadistica'], how='inner')

# Verificar y manejar valores faltantes o infinitos
if data_combinada.isna().any().any() or not np.isfinite(data_combinada).all().all():
    print("Se encontraron valores NaN o infinitos en los datos. Eliminando filas problemáticas...")
    data_combinada = data_combinada.dropna().replace([np.inf, -np.inf], 0)

# Crear variables X (predictoras) y y (objetivo)
X = data_combinada[['PM25']]  # Usar calidad del aire (PM2.5) como predictor
y = data_combinada['NumTotal']

# Dividir en conjunto de entrenamiento (70%) y prueba (30%)
train_size = int(0.7 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Verificar dimensiones y valores en los conjuntos de entrenamiento y prueba
if X_train.empty or y_train.empty:
    raise ValueError("El conjunto de entrenamiento está vacío después de filtrar los datos.")
if not np.isfinite(X_train).all().all() or not np.isfinite(y_train).all():
    raise ValueError("El conjunto de entrenamiento contiene valores inválidos después de filtrar.")

# Ajustar modelo SARIMA
sarima_model = SARIMAX(y_train, exog=X_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
sarima_result = sarima_model.fit(disp=False)

# Predicción en el conjunto de prueba
y_pred = sarima_result.forecast(steps=len(X_test), exog=X_test)

# Evaluar el modelo
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Visualización de los resultados
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Valores Reales", color="blue")
plt.plot(y_pred, label="Valores Predichos", color="orange")
plt.title("Valores Reales vs Predichos (SARIMA)")
plt.xlabel("Semana")
plt.ylabel("Urgencias")
plt.legend()
plt.show()

