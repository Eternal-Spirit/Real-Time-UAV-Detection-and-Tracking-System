"""
train_detect.py — Retrain standard YOLOv8s detector (RTX 5060).

Usage:
    python train_detect.py --epochs 1  --name timing_test   # timing test
    python train_detect.py --epochs 100                     # full run

After training, evaluates best.pt on the held-out TEST split
(Anti-drone-10/test) so reported numbers are not from validation.
The previous weights are backed up at models/best_old_30ep.pt.
"""

import argparse
import os
import shutil
import time

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    # int = fixed batch size; float in (0,1) = AutoBatch fraction of VRAM
    parser.add_argument('--batch', type=float, default=0.85)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cache', default='disk', choices=['disk', 'ram', 'none'])
    parser.add_argument('--name', default='yolov8s_5060_100ep')
    parser.add_argument('--no-copy', action='store_true',
                        help='Do not copy best.pt into models/')
    args = parser.parse_args()
    batch = int(args.batch) if args.batch >= 1 else args.batch
    cache = False if args.cache == 'none' else args.cache

    model = YOLO('yolov8s.pt')

    t0 = time.time()
    model.train(
        data     = 'drone.yaml',
        epochs   = args.epochs,
        imgsz    = 640,
        device   = 0,
        batch    = batch,
        amp      = True,
        workers  = args.workers,
        cache    = cache,
        cos_lr   = True,
        patience = 20,
        project  = os.path.abspath('runs/train'),
        name     = args.name,
        exist_ok = True,
        plots    = True,
    )
    elapsed = time.time() - t0
    print(f'\n[TIME] Training took {elapsed/60:.1f} min '
          f'({elapsed/args.epochs:.1f} s/epoch)')

    run_dir = os.path.join('runs', 'train', args.name)
    best = os.path.join(run_dir, 'weights', 'best.pt')

    # ── Held-out test-set evaluation ──────────────────────────
    print('\n[TEST] Evaluating best.pt on held-out test split...')
    test_model = YOLO(best)
    m = test_model.val(data='drone.yaml', split='test', imgsz=640,
                       batch=16, device=0, workers=0,
                       project=os.path.abspath('runs/test'),
                       name=args.name, exist_ok=True)
    report = (
        f'Run: {args.name}\n'
        f'Epochs requested: {args.epochs}\n'
        f'Training time: {elapsed/60:.1f} min\n'
        f'TEST precision : {m.box.mp:.4f}\n'
        f'TEST recall    : {m.box.mr:.4f}\n'
        f'TEST mAP@0.50  : {m.box.map50:.4f}\n'
        f'TEST mAP@50-95 : {m.box.map:.4f}\n'
    )
    print('\n' + report)
    with open(os.path.join(run_dir, 'test_metrics.txt'), 'w') as f:
        f.write(report)

    if not args.no_copy:
        os.makedirs('models', exist_ok=True)
        shutil.copy(best, 'models/best.pt')
        print('[OK] Copied best.pt -> models/best.pt '
              '(old weights at models/best_old_30ep.pt)')


if __name__ == '__main__':
    main()
