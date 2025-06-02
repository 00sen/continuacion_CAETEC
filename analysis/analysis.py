#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import os

PREDS        = "RESULTADO_CSV.csv"
GROUND_TRUTH = "ground_truth.csv"

def parse_beds_str(beds_str: str) -> set[int]:
    if pd.isna(beds_str) or str(beds_str).strip() == "":
        return set()
    return set(int(x) for x in str(beds_str).split(",") if x.strip().isdigit())

# ----------------------------------------------------------------------
# 1. Carga de datos
# ----------------------------------------------------------------------
df_pred = pd.read_csv(PREDS)            # columnas: filename, n_cows, beds
df_gt   = pd.read_csv(GROUND_TRUTH)      # columnas: filename, n_cows, beds

# Renombramos columnas del ground truth para evitar colisiones
df_gt = df_gt.rename(columns={"n_cows": "true_n_cows", "beds": "true_beds"})

# ----------------------------------------------------------------------
# 2. Merge de predicciones y ground truth
# ----------------------------------------------------------------------
df = pd.merge(df_gt, df_pred, on="filename", how="inner", validate="one_to_one")

# ----------------------------------------------------------------------
# 3. Cálculo de métricas por fila
# ----------------------------------------------------------------------
err_counts           = []
correct_count_flags  = []
correct_beds_flags   = []

for idx, row in df.iterrows():
    # Conteo de vacas
    true_nc = int(row["true_n_cows"])
    pred_nc = int(row["n_cows"])
    err_counts.append(abs(pred_nc - true_nc))
    correct_count_flags.append(int(pred_nc == true_nc))

    # Camas
    true_beds = parse_beds_str(row["true_beds"])
    pred_beds = parse_beds_str(row["beds"])
    correct_beds_flags.append(int(pred_beds == true_beds))

# Añadimos columnas al DataFrame
df["err_count"]     = err_counts
df["correct_count"] = correct_count_flags
df["correct_beds"]  = correct_beds_flags

# ----------------------------------------------------------------------
# 4. Estadísticos globales
# ----------------------------------------------------------------------
N               = len(df)
mae_count       = np.mean(err_counts)
accuracy_count  = np.mean(correct_count_flags) * 100
accuracy_beds   = np.mean(correct_beds_flags)  * 100

summary_dict = {
    "N imágenes"                  : N,
    "MAE conteo vacas"            : round(mae_count, 3),
    "Exactitud conteo (%)"        : round(accuracy_count, 2),
    "Exactitud camas (%)"         : round(accuracy_beds, 2),
}

# Imprimimos resumen en consola
print("=== RESUMEN GLOBAL ===")
for k, v in summary_dict.items():
    print(f"{k:<25}: {v}")
print()
print(df)

# ----------------------------------------------------------------------
# 5. Generación de gráficos con matplotlib
# ----------------------------------------------------------------------
# 5.1 Histograma de errores de conteo (err_count)
plt.figure(figsize=(8, 5))
plt.hist(err_counts, bins=range(0, max(err_counts) + 2), edgecolor="black", align="left")
plt.title("Distribución del error absoluto en conteo de vacas")
plt.xlabel("Error en número de vacas")
plt.ylabel("Frecuencia")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("histograma_error_conteo.png")
plt.close()  # cerramos la figura para liberar memoria

# 5.2 Gráfico de barras con exactitudes (conteo vs camas)
labels = ["Conteo de vacas", "Camas predichas"]
values = [accuracy_count, accuracy_beds]

plt.figure(figsize=(6, 4))
plt.bar(labels, values, edgecolor="black")
plt.ylim(0, 100)
for i, v in enumerate(values):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
plt.title("Exactitud (%) en conteo y detección de camas")
plt.ylabel("Exactitud (%)")
plt.tight_layout()
plt.savefig("exactitud_conteo_camas.png")
plt.close()

# 5.3 Si quieres, también puedes guardar el DataFrame completo con métricas en un CSV separado
output_metrics = "metricas_detalladas.csv"
df.to_csv(output_metrics, index=False)
print(f"\nSe han creado los siguientes archivos para visualización:\n"
      f"  1) histograma_error_conteo.png\n"
      f"  2) exactitud_conteo_camas.png\n"
      f"  3) {output_metrics} (DataFrame con columnas: filename, true_n_cows, true_beds, n_cows, beds, err_count, correct_count, correct_beds)")
