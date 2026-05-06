# ============================================================
#  preprocessing.py  —  CLAHE frame enhancement
# ============================================================

import cv2
import numpy as np
from config import CLAHE_ENABLED, CLAHE_CLIP, CLAHE_GRID


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE to the L channel of LAB colour space.
    Boosts contrast in low-light/hazy frames without colour shift.
    """
    if not CLAHE_ENABLED:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Master preprocessing pipeline — call this once per frame
    before passing to YOLOv8.
    Add more steps here in future phases.
    """
    frame = enhance_frame(frame)
    return frame


# ── Visual test — run directly to preview CLAHE on an image ──
# python preprocessing.py --image path\to\dark_frame.jpg

if __name__ == '__main__':
    import sys, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to a dark/hazy test frame')
    parser.add_argument('--save', default='enhanced_preview.jpg')
    args = parser.parse_args()

    original = cv2.imread(args.image)
    if original is None:
        print(f'[ERROR] Could not read: {args.image}')
        sys.exit(1)

    enhanced = enhance_frame(original)
    h, w = original.shape[:2]
    divider = np.full((h, 4, 3), 200, dtype=np.uint8)
    comparison = np.hstack([original, divider, enhanced])
    cv2.putText(comparison, 'Original',       (10,    30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.putText(comparison, 'CLAHE Enhanced', (w+14,  30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.imwrite(args.save, comparison)

    print(f'[OK]  Saved comparison → {args.save}')
    print(f'      Brightness before : {original.mean():.1f}')
    print(f'      Brightness after  : {enhanced.mean():.1f}')
