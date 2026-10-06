from ultralytics import YOLO
import os

MODEL_PATH = "best.pt"

print("=" * 60)
print("LOADING TRAINED YOLO11 MODEL")
print("=" * 60)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "best.pt was not found in the current folder."
    )

model = YOLO(MODEL_PATH)

print("\nModel loaded successfully.")

print("\nModel file:")
print(MODEL_PATH)

print("\nModel size:")
print(
    round(os.path.getsize(MODEL_PATH) / (1024 * 1024), 2),
    "MB"
)

print("\nClass names:")
print(model.names)

print("\nNumber of classes:")
print(len(model.names))

print("\nModel architecture information:")
print("-" * 60)

model.info()

print("=" * 60)
print("MODEL INSPECTION COMPLETED")
print("=" * 60)