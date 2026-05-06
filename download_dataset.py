# ============================================================
#  download_dataset.py
#  Downloads the Anti-Drone dataset from Roboflow
#  in YOLOv8 format — includes drone.yaml automatically
#
#  Run: python download_dataset.py
#  Needs: pip install roboflow
# ============================================================
 
from roboflow import Roboflow
import os
import shutil
 
API_KEY = "U963fwpEkJDUP07NThvi"


rf = Roboflow(api_key=API_KEY)
 
project = rf.workspace("drone-detection-fn0bd").project("anti-drone")
version  = project.version(10)
dataset  = version.download("yolov8")
 
print("\n" + "=" * 50)
print("  Dataset downloaded successfully!")
print("=" * 50)
print(f"\n  Location : {dataset.location}")

# Move and rename the data.yaml to drone.yaml in the current directory
yaml_path = os.path.join(dataset.location, "data.yaml")
if os.path.exists(yaml_path):
    shutil.copy(yaml_path, "drone.yaml")
    print(f"  Copied data.yaml to drone.yaml")
else:
    print(f"  Could not find data.yaml in {dataset.location}")
