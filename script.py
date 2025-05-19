#!/usr/bin/env python3
"""
Sort images by number of cows detected with a YOLOv5 `.pt` checkpoint.
Usage:
    python3 script.py /path/to/images --weights model.pt --device cuda:0
"""

# ----------------------------------------------------------------------
# 0. WindowsPath hot-patch (needed when loading a checkpoint saved on Windows)
# ----------------------------------------------------------------------
import pathlib, sys
if sys.platform != "win32":           # we're on Linux/macOS
    pathlib.WindowsPath = pathlib.PosixPath  # safe monkey-patch

# ----------------------------------------------------------------------
# 1. Standard libs and YOLOv5
# ----------------------------------------------------------------------
import argparse, os, shutil, cv2, torch, yolov5

# ----------------------------------------------------------------------
# 2. Command-line interface
# ----------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Sort images into folders by cow count")
    ap.add_argument("input_folder", help="Folder containing images")
    ap.add_argument("--weights", default="model.pt", help="Path to YOLOv5 .pt file")
    ap.add_argument("--output",  default="by_cow_count", help="Destination root folder")
    ap.add_argument("--device",  default="cpu", help="cpu | cuda | cuda:0 | cuda:1 ...")
    ap.add_argument("--imgsz",   type=int, default=640, help="Inference image size")
    return ap.parse_args()

# ----------------------------------------------------------------------
# 3. Main logic
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    # --- Locate installed yolov5 package directory and load model ----------
    repo_dir = os.path.dirname(yolov5.__file__)
    model = torch.hub.load(
        repo_dir, 'custom',
        path=args.weights,
        source='local',
        device=args.device
    )

    # --- Figure out which class IDs mean "cow" -----------------------------
    if isinstance(model.names, dict):         # {id: name}
        cow_ids = [cid for cid, name in model.names.items()
                   if str(name).lower() == "cow"]
    else:                                     # list/tuple
        cow_ids = [i for i, name in enumerate(model.names)
                   if str(name).lower() == "cow"]
    if not cow_ids:                           # single-class model fallback
        cow_ids = [0]
    print(f"Cow class-ID(s): {cow_ids}")

    # --- Create destination root
    os.makedirs(args.output, exist_ok=True)

    # --- Walk through every image file -------------------------------------
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    for file in os.listdir(args.input_folder):
        if not file.lower().endswith(valid_exts):
            continue
        src_path = os.path.join(args.input_folder, file)

        # Inference
        results = model(src_path, size=args.imgsz)
        preds = results.xyxy[0]               # [n,6] tensor: x1 y1 x2 y2 conf cls

        # Count cows
        n_cows = sum(int(p[5]) in cow_ids for p in preds)

        # Destination sub-folder like 00, 01, 02 …
        dst_sub = os.path.join(args.output, f"{n_cows:02d}")
        os.makedirs(dst_sub, exist_ok=True)

        # Save annotated copy
        results.render()                      # draws boxes in-place
        cv2.imwrite(os.path.join(dst_sub, f"bb_{file}"), results.imgs[0])

        # Copy original
        shutil.copy2(src_path, os.path.join(dst_sub, file))

        print(f"{file:30s}  →  {n_cows} cow(s)")

    print("✓ All done!")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
