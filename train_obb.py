"""
train_obb.py — Retrain genuine YOLOv8s-OBB model using SAM-generated rotated labels.

Usage:
    python train_obb.py --epochs 60
"""

import os
import time
import shutil
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch', type=float, default=0.85)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--name', default='yolov8s_obb_5060_run')
    args = parser.parse_args()

    # Backup previous obb_best.pt
    if os.path.exists('models/obb_best.pt') and not os.path.exists('models/obb_best_old.pt'):
        shutil.copy('models/obb_best.pt', 'models/obb_best_old.pt')
        print("[OK] Backed up old OBB model to models/obb_best_old.pt")

    batch = int(args.batch) if args.batch >= 1 else args.batch
    model = YOLO('yolov8s-obb.pt')

    t0 = time.time()
    results = model.train(
        data     = 'drone_obb.yaml',
        epochs   = args.epochs,
        imgsz    = 640,
        device   = 0,
        batch    = batch,
        amp      = True,
        workers  = args.workers,
        cache    = 'disk',
        cos_lr   = True,
        patience = 15,
        project  = os.path.abspath('runs/obb'),
        name     = args.name,
        exist_ok = True,
        plots    = True,
    )
    elapsed = time.time() - t0
    print(f"\n[TIME] OBB Training finished in {elapsed/60:.1f} mins.")

    run_dir = os.path.join('runs', 'obb', args.name)
    best_weights = os.path.join(run_dir, 'weights', 'best.pt')

    # Evaluate on held-out test split
    print("\n[TEST] Evaluating best OBB model on held-out test split...")
    test_model = YOLO(best_weights)
    m = test_model.val(
        data='drone_obb.yaml',
        split='test',
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        project=os.path.abspath('runs/test_obb'),
        name=args.name,
        exist_ok=True,
    )

    metric_obj = getattr(m, 'obb', getattr(m, 'box', None))
    p = metric_obj.mp if metric_obj is not None else 0.0
    r = metric_obj.mr if metric_obj is not None else 0.0
    map50 = metric_obj.map50 if metric_obj is not None else 0.0
    map95 = metric_obj.map if metric_obj is not None else 0.0

    report = (
        f"Run: {args.name}\n"
        f"Epochs: {args.epochs}\n"
        f"Training time: {elapsed/60:.1f} min\n"
        f"TEST OBB Precision : {p:.4f}\n"
        f"TEST OBB Recall    : {r:.4f}\n"
        f"TEST OBB mAP@0.50  : {map50:.4f}\n"
        f"TEST OBB mAP@50-95 : {map95:.4f}\n"
    )
    print("\n" + report)
    with open(os.path.join(run_dir, 'obb_test_metrics.txt'), 'w') as f:
        f.write(report)

    # Copy to models/obb_best.pt
    os.makedirs('models', exist_ok=True)
    shutil.copy(best_weights, 'models/obb_best.pt')
    print(f"[OK] Copied best OBB weights -> models/obb_best.pt")


if __name__ == '__main__':
    main()
