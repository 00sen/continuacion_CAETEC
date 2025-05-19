#!/usr/bin/env python3
"""
Sort images by number of cows detected with a YOLOv5 checkpoint.

Example:
    python3 script.py images --weights model.pt            # CPU
    python3 script.py images --weights model.pt --device cuda:0  # GPU
"""

# ----------------------------------------------------------------------
# 0. Fix Windows checkpoints on Linux/macOS
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

# Suppress FutureWarnings from yolov5/models/common.py
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"yolov5\.models\.common"
)

# ----------------------------------------------------------------------
# 2. CLI
# ----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="Sort images into folders by cow count")
    p.add_argument("input_folder", help="Folder containing images")
    p.add_argument("--weights", default="model.pt", help="YOLOv5 .pt file")
    p.add_argument("--output",  default="by_cow_count", help="Destination root folder")
    p.add_argument("--device",  default="cpu", help="cpu | cuda | cuda:0 ...")
    p.add_argument("--imgsz",   type=int, default=640, help="Inference image size")
    return p.parse_args()

# ----------------------------------------------------------------------
# 3. Annotate helper
# ----------------------------------------------------------------------
def draw_boxes(img, preds, cow_ids: List[int]):
    for *xyxy, conf, cls in preds.tolist():
        if int(cls) not in cow_ids:
            continue
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "cow", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img

# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------
def main():
    args = get_args()

    # >>> Load YOLOv5 model from installed package
    repo_dir = os.path.dirname(yolov5.__file__)
    model = torch.hub.load(repo_dir, 'custom',
                           path=args.weights, source='local',
                           device=args.device)

    # >>> Determine class IDs for "cow"
    if isinstance(model.names, dict):   # {id: name}
        cow_ids = [i for i, n in model.names.items() if str(n).lower() == "cow"]
    else:                               # list/tuple
        cow_ids = [i for i, n in enumerate(model.names) if str(n).lower() == "cow"]
    if not cow_ids:
        cow_ids = [0]                   # fallback single-class
    # print(f"Cow class-ID(s): {cow_ids}")

    os.makedirs(args.output, exist_ok=True)
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    skipped = []

    print("Work in progress...")
    # >>> Iterate through images
    for fname in os.listdir(args.input_folder):
        if not fname.lower().endswith(valid_ext):
            continue
        src = os.path.join(args.input_folder, fname)

        # safe inference
        try:
            results = model(src, size=args.imgsz)
            preds = results.xyxy[0]            # tensor [n,6]
        except Exception as e:
            print(f"⚠️  Skipping {fname}  ({e})")
            skipped.append(fname)
            continue

        n_cows = sum(int(p[5]) in cow_ids for p in preds)
        dst = os.path.join(args.output, f"{n_cows:02d}")
        os.makedirs(dst, exist_ok=True)

        # annotate and save
        img = cv2.imread(src)
        img = draw_boxes(img, preds, cow_ids)
        cv2.imwrite(os.path.join(dst, f"bb_{fname}"), img)
        shutil.copy2(src, os.path.join(dst, fname))
        # print(f"{fname:30s}  →  {n_cows} cow(s)")

    # >>> Summary
    print("\n✓ Finished!")
    if skipped:
        print(f"Skipped {len(skipped)} unreadable file(s):")
        for s in skipped:
            print("   •", s)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
