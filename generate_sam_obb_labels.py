"""
generate_sam_obb_labels.py — Converts standard axis-aligned drone labels
into genuine Oriented Bounding Box (OBB) labels using FastSAM and cv2.minAreaRect.

Usage:
    python generate_sam_obb_labels.py --split train
    python generate_sam_obb_labels.py --split valid
    python generate_sam_obb_labels.py --split test
    python generate_sam_obb_labels.py --all
"""

import os
import glob
import time
import argparse
import cv2
import numpy as np
from ultralytics import FastSAM


def process_split(split, model, min_size_px=14):
    print(f"\n==========================================")
    print(f" Processing split: {split}")
    print(f"==========================================")
    
    base_dir = os.path.join('Anti-drone-10', split)
    img_dir = os.path.join(base_dir, 'images')
    
    # Check for labels_std first, fallback to labels
    src_label_dir = os.path.join(base_dir, 'labels_std')
    if not os.path.exists(src_label_dir):
        src_label_dir = os.path.join(base_dir, 'labels')
        
    dst_label_dir = os.path.join(base_dir, 'labels_obb')
    os.makedirs(dst_label_dir, exist_ok=True)
    
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    total_imgs = len(img_files)
    print(f"Found {total_imgs} images in {img_dir}")
    print(f"Reading labels from: {src_label_dir}")
    print(f"Saving OBB labels to: {dst_label_dir}")
    
    t0 = time.time()
    converted_boxes = 0
    tilted_boxes = 0
    
    for idx, img_path in enumerate(img_files):
        fname = os.path.splitext(os.path.basename(img_path))[0]
        lbl_file = os.path.join(src_label_dir, f"{fname}.txt")
        out_file = os.path.join(dst_label_dir, f"{fname}.txt")
        
        if not os.path.exists(lbl_file):
            with open(out_file, 'w') as f:
                pass
            continue
            
        with open(lbl_file, 'r') as f:
            lines = f.read().strip().splitlines()
            
        if not lines:
            with open(out_file, 'w') as f:
                pass
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        obb_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = parts[0]
            cx, cy, bw, bh = [float(x) for x in parts[1:5]]
            
            box_w_px = bw * w
            box_h_px = bh * h
            
            x1 = max(0, int((cx - bw / 2) * w))
            y1 = max(0, int((cy - bh / 2) * h))
            x2 = min(w - 1, int((cx + bw / 2) * w))
            y2 = min(h - 1, int((cy + bh / 2) * h))
            
            # Default horizontal box fallback
            default_pts = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ], dtype=np.float32)
            pts = default_pts
            
            # If target has discernible shape (>= min_size_px), run FastSAM
            if box_w_px >= min_size_px and box_h_px >= min_size_px and (x2 > x1) and (y2 > y1):
                try:
                    res = model(img, bboxes=[[x1, y1, x2, y2]], device=0, verbose=False)
                    if res[0].masks is not None and len(res[0].masks.data):
                        m = res[0].masks.data[0].cpu().numpy().astype(np.uint8)
                        m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        # Crop mask strictly to bbox area
                        crop_mask = np.zeros_like(m_resized)
                        crop_mask[y1:y2, x1:x2] = m_resized[y1:y2, x1:x2]
                        
                        cnts, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            c = max(cnts, key=cv2.contourArea)
                            if cv2.contourArea(c) > 20:
                                rect = cv2.minAreaRect(c)
                                box_pts = cv2.boxPoints(rect)
                                angle = rect[2]
                                # Check if non-zero angle
                                if abs(angle) > 2.0 and abs(angle - 90.0) > 2.0:
                                    pts = box_pts
                                    tilted_boxes += 1
                except Exception as e:
                    pts = default_pts
                    
            converted_boxes += 1
            # Normalize 4 points to 0.0 - 1.0
            norm_pts = []
            for p in pts:
                px = np.clip(p[0] / w, 0.0, 1.0)
                py = np.clip(p[1] / h, 0.0, 1.0)
                norm_pts.append(f"{px:.6f} {py:.6f}")
                
            obb_lines.append(f"{cls_id} {' '.join(norm_pts)}")
            
        with open(out_file, 'w') as f:
            f.write('\n'.join(obb_lines) + ('\n' if obb_lines else ''))
            
        if (idx + 1) % 250 == 0 or (idx + 1) == total_imgs:
            elapsed = time.time() - t0
            fps = (idx + 1) / max(elapsed, 1e-5)
            print(f"[{split}] {idx+1}/{total_imgs} images processed ({fps:.1f} img/s) | Tilted boxes: {tilted_boxes}/{converted_boxes}")
            
    print(f"\nSplit '{split}' done in {(time.time()-t0)/60:.1f} mins.")
    print(f"Total boxes: {converted_boxes}, Tilted rotated boxes extracted: {tilted_boxes} ({tilted_boxes/max(converted_boxes,1)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='valid', choices=['train', 'valid', 'test'])
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    
    print("Loading FastSAM model onto GPU 0...")
    model = FastSAM('FastSAM-s.pt')
    
    splits = ['valid', 'test', 'train'] if args.all else [args.split]
    for s in splits:
        process_split(s, model)
        
    print("\n==========================================")
    print(" All requested splits completed!")
    print("==========================================")


if __name__ == '__main__':
    main()
