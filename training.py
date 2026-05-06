from ultralytics import YOLO
import shutil, os

def train_obb():
    model = YOLO('yolov8s-obb.pt')   # pretrained OBB backbone

    results = model.train(
        data    = 'drone_obb.yaml',
        epochs  = 30,
        imgsz   = 640,
        device  = 0,               # GPU 0 (RTX 4060)
        batch   = 4,               # Reduced to 4
        workers = 0,               # Prevent PyTorch deadlock on Windows
        amp     = False,           # Prevent CUDA unknown error during AMP checks
        patience= 10,              # early stopping
        project = 'runs/obb',
        name    = 'train',
        exist_ok= True,
    )

    best_weights = 'runs/obb/train/weights/best.pt'
    dst = 'models/obb_best.pt'
    os.makedirs('models', exist_ok=True)
    shutil.copy(best_weights, dst)
    print(f"Training complete. Weights saved to: {dst}")
    print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")

if __name__ == '__main__':
    train_obb()
