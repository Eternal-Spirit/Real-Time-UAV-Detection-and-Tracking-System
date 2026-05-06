import os
import cv2
import pandas as pd
import numpy as np
from stats import plot_model_training

def main():
    stats_dir = 'stats'
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    print("Generating training curves from results.csv...")
    training_csv = 'runs/detect/runs/train/yolov8s_clahe_v12/results.csv'
    training_img = plot_model_training(training_csv)
    
    if training_img is not None and training_img.any():
        # Convert RGB to BGR for saving with cv2
        bgr = cv2.cvtColor(training_img, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(stats_dir, 'training_curves.png')
        cv2.imwrite(out_path, bgr)
        print(f"Saved training curves to {out_path}")
    else:
        print("Failed to generate training curves. Ensure results.csv exists.")

    log_path = 'outputs/intrusion_log.csv'
    if os.path.exists(log_path):
        print("Generating intrusion log summary...")
        df = pd.read_csv(log_path)
        summary_path = os.path.join(stats_dir, 'intrusion_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Drone Detection System - Intrusion Summary\n")
            f.write("========================================\n\n")
            f.write(f"Total Detections Logged: {len(df)}\n")
            
            if 'zone' in df.columns:
                f.write("\nBreakdown by Threat Zone:\n")
                f.write("-" * 25 + "\n")
                f.write(df['zone'].value_counts().to_string() + "\n")
                
            if 'drone_type' in df.columns:
                f.write("\nBreakdown by Drone Type:\n")
                f.write("-" * 25 + "\n")
                f.write(df['drone_type'].value_counts().to_string() + "\n")
                
        print(f"Saved intrusion summary to {summary_path}")
    else:
        print(f"Intrusion log not found at {log_path}, skipping summary.")

    print("\nStats generation complete!")

if __name__ == "__main__":
    main()
