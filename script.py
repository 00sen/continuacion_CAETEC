#!/usr/bin/env python3
# ----------------------------------------------------------------------
# 0. Ajuste Windows ↔ Linux
# ----------------------------------------------------------------------
import pathlib, sys
if sys.platform != "win32":            # ejecutándose en POSIX
    pathlib.WindowsPath = pathlib.PosixPath     # type: ignore

# ----------------------------------------------------------------------
# 1. Imports
# ----------------------------------------------------------------------
import argparse, os, shutil, csv, cv2, torch, yolov5
from typing import List, Union
import warnings

# Función para ignorar advertencias innecesarias
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"yolov5\.models\.common"
)

# ----------------------------------------------------------------------
# 2. Parámetros de línea de comandos
# Aquí se definen las opciones que el usuario puede tomar a la hora de correr el script
# ----------------------------------------------------------------------
def get_args():
    """
    input_folder : carpeta con imágenes (obligatorio)
    --model      : ruta al modelo YOLOv5 (.pt)
    --format     : images (default) o csv
    """
    p = argparse.ArgumentParser(
        description="Clasifica imágenes según la cantidad de vacas detectadas")
    p.add_argument("input_folder", help="Carpeta con las imágenes a procesar")
    p.add_argument("--model",  default="modelBeds.pt", help="Archivo .pt del modelo")
    p.add_argument("--format", choices=["images", "csv"], default="csv",
                   help="Tipo de salida: ‘images’ o ‘csv’")
    return p.parse_args()

# ----------------------------------------------------------------------
# 3. Calibración manual de divisores de cama
# Aquí se eligen los pixeles con los cuales están divididas las camas,
# se puso la primera cama a 260 pixeles porque esta casi no se ve,
# la segunda a 720 pixeles y la tercera a partir de los 1200 pixeles
# ----------------------------------------------------------------------
DIVS_X = [260, 720, 1200]   # Pixeles de las camas de izquierda a derecha

def bed_id_for_bbox(x1, y1, x2, y2, divs_x=DIVS_X):
    """Devuelve 0-3 según dónde caiga el centro del bbox."""
    cx = (x1 + x2) / 2
    for i, div in enumerate(divs_x):
        if cx < div:
            return i
    return len(divs_x)      # regresa la última cama

# ----------------------------------------------------------------------
# 4. Auxiliar: si la carpeta ya existe la sobreescribe
# De esta manera si la carpeta RESULTADO_IMAGENES ya existe esta se borra antes de comenzar
# ----------------------------------------------------------------------
def ensure_empty_dir(path: Union[str, pathlib.Path]):
    path = pathlib.Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 5. Auxiliar: dibuja las cajas de las vacas detectadas
# Si se eligió la opción (images) se pintará un recuadro verde a la vaca
# ----------------------------E-----------------------------------------
def draw_boxes(img, preds, cow_ids: List[int]):
    """Dibuja rectángulos verdes y la palabra ‘cow’."""
    for x1, y1, x2, y2, conf, cls in preds.tolist():
        cls = int(cls)
        if cls not in cow_ids:
            continue
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "cow", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img

# ----------------------------------------------------------------------
# 6. Función principal
# Solo corre el programa completo, usando las funciones ya mencionadas
# ----------------------------------------------------------------------
def main():
    args = get_args()

    # Carga del modelo
    repo_dir = os.path.dirname(yolov5.__file__)
    model = torch.hub.load(repo_dir, "custom",
                           path=args.model, source="local", device="cpu")

    # IDs de la clase “cow”
    cow_ids = ([i for i, n in model.names.items() if str(n).lower() == "cow"]
               if isinstance(model.names, dict)
               else [i for i, n in enumerate(model.names) if str(n).lower() == "cow"])
    if not cow_ids:
        cow_ids = [0]

    # Acepta archivos .png, .jpg y .jpeg
    valid_ext = (".png", ".jpg", ".jpeg")
    skipped: list[str] = []
    records: list[tuple[str, int, str]] = []   # ahora incluye camas

    # Gestionamos carpeta de salida solo si es formato imágenes
    output_dir = "RESULTADO_IMAGENES"
    if args.format == "images":
        ensure_empty_dir(output_dir)

    print("Work in progress…")
    for fname in os.listdir(args.input_folder):
        if not fname.lower().endswith(valid_ext):
            continue
        src = os.path.join(args.input_folder, fname)

        # Se aplica el modelo a la imagen actual
        # Si se encuentra algún error como un archivo corrupto u otra complicación se omite la imagen y sigue el script
        try:
            results = model(src)        
            preds = results.xyxy[0]
        except Exception as e:
            print(f"⚠️  Skipping {fname}  ({e})")
            skipped.append(fname)
            continue

        # Conteo y camas ocupadas
        n_cows = 0
        beds_used: set[int] = set()
        for x1, y1, x2, y2, conf, cls in preds.tolist():
            if int(cls) not in cow_ids:
                continue
            n_cows += 1
            if args.format == "csv":                 # solo calculamos cama si se pide csv
                beds_used.add(bed_id_for_bbox(x1, y1, x2, y2))

        # Guardamos resultado según formato
        if args.format == "images":
            dst_sub = os.path.join(output_dir, f"{n_cows:02d}")
            os.makedirs(dst_sub, exist_ok=True)
            img = cv2.imread(src)
            img = draw_boxes(img, preds, cow_ids)
            cv2.imwrite(os.path.join(dst_sub, f"bb_{fname}"), img)
        else:  # csv
            beds_str = ",".join(map(str, sorted(beds_used))) if beds_used else ""
            records.append((fname, n_cows, beds_str))

    # Salida CSV
    if args.format == "csv":
        csv_path = "RESULTADO_CSV.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "n_cows", "beds"])
            writer.writerows(records)
        print(f"\n✓ CSV guardado en {csv_path}")

    # Resumen
    print("\n✓ Finished!")
    if skipped:
        print(f"Skipped {len(skipped)} unreadable file(s):")
        for s in skipped:
            print("   •", s)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
