#!/usr/bin/env python3

# ----------------------------------------------------------------------
# 0. Arreglar cambio de windows a linux
# ----------------------------------------------------------------------
import pathlib, sys
if sys.platform != "win32":            # running on POSIX
    pathlib.WindowsPath = pathlib.PosixPath     # type: ignore

# ----------------------------------------------------------------------
# 1. Imports
# ----------------------------------------------------------------------
import argparse, os, shutil, cv2, torch, yolov5
from typing import List

import warnings
import re

# Suprimimos cualquier advertencia durante la ejecución del programa
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"yolov5\.models\.common"
)

# ----------------------------------------------------------------------
# 2. Inputs del usuario mediante la consola
# ----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="Ordenar imágenes por cantidad de vacas")
    p.add_argument("input_folder", help="Folder con las imágenes a clasificar")
    p.add_argument("--weights", default="model.pt", help="Archivo del model, terminación .pt")
    p.add_argument("--output",  default="by_cow_count", help="Folder destino")
    p.add_argument("--device",  default="cpu", help="cpu | cuda | cuda:0 ...")
    p.add_argument("--imgsz",   type=int, default=640, help="Inference image size")
    return p.parse_args()

# ----------------------------------------------------------------------
# 3. Ayudante, dibuja las cajas a las vacas encontradas
# ----------------------------------------------------------------------
def draw_boxes(img, preds, cow_ids: List[int]):
    for *xyxy, conf, cls in preds.tolist():
        if int(cls) not in cow_ids:
            continue # Ignoramos las detecciones que no sean vacas
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # Dibuja rectángulo verde
        cv2.putText(img, "cow", (x1, y1 - 6), # Escribe palabra "vaca" encima del rectánculo
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img

# ----------------------------------------------------------------------
# 4. Función main
# ----------------------------------------------------------------------
def main():
    args = get_args()

    # Cargamos el modelo YOLOv5f
    repo_dir = os.path.dirname(yolov5.__file__)
    model = torch.hub.load(repo_dir, 'custom',
                           path=args.weights, source='local',
                           device=args.device)

    # Determina qué clases del modelo son "vaca"
    if isinstance(model.names, dict):   # Si los nombres están en formato {id: nombre}
        cow_ids = [i for i, n in model.names.items() if str(n).lower() == "cow"]
    else: # Si están en una lista simple
        cow_ids = [i for i, n in enumerate(model.names) if str(n).lower() == "cow"]
    if not cow_ids:
        cow_ids = [0] # Si no encuentra "cow" asume que solo hay una clase: vacas

    os.makedirs(args.output, exist_ok=True) # Crea carpeta destino si no existe
    valid_ext = (".png", ".jpg", ".jpeg") # Tipos de archivos válidos
    skipped = [] # Aquí guardamos imágenes que fallan, ya sean corruptas u otra cosa

    print("Work in progress...")
    # Iteramos por las imágenes
    for fname in os.listdir(args.input_folder):
        if not fname.lower().endswith(valid_ext):
            continue # Ignora archivos que no sean imágenes
        src = os.path.join(args.input_folder, fname)

        # Intenta pasar la imagen por el modelo
        try:
            results = model(src, size=args.imgsz) # Ejecuta el modelo sobre la imagen
            preds = results.xyxy[0]            # Lista de recuadros detectados
        # Cacha el error y lo añade a la lista de imágenes que fallaron
        except Exception as e:
            print(f"⚠️  Skipping {fname}  ({e})")
            skipped.append(fname)
            continue

        # Cuenta cuántas vacas fueron detectadas
        n_cows = sum(int(p[5]) in cow_ids for p in preds)
        
        # Crea subcarpeta para esa cantidad de vacas
        dst = os.path.join(args.output, f"{n_cows:02d}")
        os.makedirs(dst, exist_ok=True)

        # Dibuja recuadros en una copia de la imagen
        img = cv2.imread(src)
        img = draw_boxes(img, preds, cow_ids)
        cv2.imwrite(os.path.join(dst, f"bb_{fname}"), img) # Guarda la imagen con recuadros

    # Resúmen de operación
    print("\n✓ Finished!")
    if skipped:
        print(f"Skipped {len(skipped)} unreadable file(s):")
        for s in skipped:
            print("   •", s)

# Ejecución del programa
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
