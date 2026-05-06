import os
import argparse

def convert_file(src_path, dst_path):
    with open(src_path) as f:
        lines = f.read().strip().splitlines()
    out = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, cx_str, cy_str, w_str, h_str = parts[:5]
        cx, cy, w, h = float(cx_str), float(cy_str), float(w_str), float(h_str)
        
        # Calculate 4 corners of axis-aligned box
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy - h / 2
        x3 = cx + w / 2
        y3 = cy + h / 2
        x4 = cx - w / 2
        y4 = cy + h / 2
        
        # YOLO OBB format: cls x1 y1 x2 y2 x3 y3 x4 y4
        out.append(f"{cls} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}")
    with open(dst_path, 'w') as f:
        f.write('\n'.join(out))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='Input labels directory')
    parser.add_argument('--output', required=True, help='Output labels directory')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    files = [f for f in os.listdir(args.input) if f.endswith('.txt')]
    for fname in files:
        convert_file(
            os.path.join(args.input,  fname),
            os.path.join(args.output, fname)
        )
    print(f"Converted {len(files)} label files -> {args.output}")

if __name__ == '__main__':
    main()
