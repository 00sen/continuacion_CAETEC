#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

PREDS        = "RESULTADO_CSV.csv"   # CSV que genera tu modelo
GROUND_TRUTH = "ground_truth.csv"    # CSV con valores correctos

# ----------------------------------------------------------------------
# Función auxiliar
# ----------------------------------------------------------------------
def parse_beds_str(beds_str: str) -> set[int]:
    """Convierte "0,1" → {0,1}. Cadena vacía/NaN → set()."""
    if pd.isna(beds_str) or str(beds_str).strip() == "":
        return set()
    return set(int(x) for x in str(beds_str).split(",") if x.strip().isdigit())

# ----------------------------------------------------------------------
# 1. Carga de datos
# ----------------------------------------------------------------------
df_pred = pd.read_csv(PREDS)            # filename, n_cows, beds
df_gt   = pd.read_csv(GROUND_TRUTH)     # filename, n_cows, beds
df_gt = df_gt.rename(columns={"n_cows": "true_n_cows", "beds": "true_beds"})

# ----------------------------------------------------------------------
# 2. Merge predicciones ↔ ground-truth
# ----------------------------------------------------------------------
df = pd.merge(df_gt, df_pred, on="filename", how="inner", validate="one_to_one")

# ----------------------------------------------------------------------
# 3. Métricas por imagen
# ----------------------------------------------------------------------
err_counts           = []
correct_count_flags  = []
correct_beds_flags   = []
hamming_list         = []      # ← lista para Hamming Loss por imagen

for _, row in df.iterrows():
    # ---------- Conteo ----------
    true_nc = int(row["true_n_cows"])
    pred_nc = int(row["n_cows"])
    err_counts.append(abs(pred_nc - true_nc))
    correct_count_flags.append(int(pred_nc == true_nc))

    # ---------- Camas ----------
    true_beds = parse_beds_str(row["true_beds"])
    pred_beds = parse_beds_str(row["beds"])
    correct_beds_flags.append(int(pred_beds == true_beds))

    # Hamming Loss por imagen  (camas mal etiquetadas / 4)
    hamming_img = len(true_beds ^ pred_beds) / 4    # XOR
    hamming_list.append(hamming_img)

# Guardamos columnas nuevas
df["err_count"]     = err_counts
df["correct_count"] = correct_count_flags
df["correct_beds"]  = correct_beds_flags
df["hamming"]       = hamming_list

# ----------------------------------------------------------------------
# 4. Estadísticos globales
# ----------------------------------------------------------------------
N               = len(df)
mae_count       = np.mean(err_counts)
rmse_count      = np.sqrt(np.mean(np.square(err_counts)))
mape_count      = np.mean(np.array(err_counts) / (df["true_n_cows"] + 1e-9)) * 100
accuracy_count  = np.mean(correct_count_flags) * 100
accuracy_beds   = np.mean(correct_beds_flags)  * 100
hamming_loss    = np.mean(hamming_list)        * 100   # en %

summary_dict = {
    "N imágenes"             : N,
    "MAE conteo vacas"       : round(mae_count, 3),
    "RMSE conteo"            : round(rmse_count, 3),
    "MAPE conteo (%)"        : round(mape_count, 2),
    "Exactitud conteo (%)"   : round(accuracy_count, 2),
    "Exactitud camas (%)"    : round(accuracy_beds, 2),
    "Hamming Loss camas (%)" : round(hamming_loss, 2),
}

# ----------------------------------------------------------------------
# 5. Salida en consola
# ----------------------------------------------------------------------
print("=== RESUMEN GLOBAL ===")
for k, v in summary_dict.items():
    print(f"{k:<25}: {v}")
print("\nPrimeras filas del detalle:")
print(df.head())

# ----------------------------------------------------------------------
# 6. Gráficos
# ----------------------------------------------------------------------
# 6.1 Histograma del error de conteo
plt.figure(figsize=(8, 5))
plt.hist(err_counts,
         bins=range(0, max(err_counts) + 2),
         edgecolor="black", align="left")
plt.title("Distribución del error absoluto en conteo de vacas")
plt.xlabel("Error (|pred − true|)")
plt.ylabel("Frecuencia")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("histograma_error_conteo.png")
plt.close()

# 6.2 Barras con métricas principales (los valores “altos son mejores”)
labels = ["Acc. conteo", "Acc. camas", "100-Hamming"]
values = [accuracy_count, accuracy_beds, 100 - hamming_loss]

plt.figure(figsize=(7, 4))
plt.bar(labels, values, edgecolor="black")
plt.ylim(0, 100)
for i, v in enumerate(values):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
plt.title("Exactitud global (%)")
plt.ylabel("Porcentaje (%)")
plt.tight_layout()
plt.savefig("metricas_globales.png")
plt.close()

# 6.3 Guardar DataFrame con todas las métricas
df.to_csv("metricas_detalladas.csv", index=False)

print("\nSe han creado los archivos:")
print("  • histograma_error_conteo.png")
print("  • metricas_globales.png")
print("  • metricas_detalladas.csv")

