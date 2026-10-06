from ultralytics import YOLO
import os

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "best.pt"

DATA_YAML = os.path.join(
    "content",
    "waste_detection",
    "data.yaml"
)

# ============================================================
# CHECK PATHS
# ============================================================

print("=" * 70)
print("YOLO11 MODEL EVALUATION")
print("=" * 70)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(DATA_YAML):
    raise FileNotFoundError(
        f"data.yaml not found: {DATA_YAML}"
    )

print("\nModel:")
print(os.path.abspath(MODEL_PATH))

print("\nDataset configuration:")
print(os.path.abspath(DATA_YAML))

# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(MODEL_PATH)

print("\nModel loaded successfully.")

# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING VALIDATION")
print("=" * 70)

metrics = model.val(
    data=DATA_YAML,
    imgsz=640,
    batch=16,
    device="cpu",
    plots=True,
    verbose=True
)

# ============================================================
# OVERALL METRICS
# ============================================================

print("\n" + "=" * 70)
print("OVERALL MODEL PERFORMANCE")
print("=" * 70)

print(
    f"Precision : {metrics.box.mp:.4f}"
)

print(
    f"Recall    : {metrics.box.mr:.4f}"
)

print(
    f"mAP50     : {metrics.box.map50:.4f}"
)

print(
    f"mAP50-95  : {metrics.box.map:.4f}"
)

# ============================================================
# F1 SCORE
# ============================================================

precision = metrics.box.mp
recall = metrics.box.mr

if (precision + recall) > 0:
    f1 = 2 * precision * recall / (precision + recall)
else:
    f1 = 0.0

print(
    f"F1 Score  : {f1:.4f}"
)

# ============================================================
# PER-CLASS METRICS
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)

# Convert class names to strings
class_names = {
    int(k): str(v)
    for k, v in model.names.items()
}

for i in range(len(class_names)):

    class_name = class_names[i]

    class_precision = metrics.box.p[i]
    class_recall = metrics.box.r[i]
    class_map50 = metrics.box.ap50[i]
    class_map = metrics.box.ap[i]

    # Calculate class F1
    if (class_precision + class_recall) > 0:
        class_f1 = (
            2
            * class_precision
            * class_recall
            / (class_precision + class_recall)
        )
    else:
        class_f1 = 0.0

    print(
        f"{class_name:<10} | "
        f"Precision: {class_precision:.4f} | "
        f"Recall: {class_recall:.4f} | "
        f"mAP50: {class_map50:.4f} | "
        f"mAP50-95: {class_map:.4f} | "
        f"F1: {class_f1:.4f}"
    )

# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print(
    "Model file size:",
    round(
        os.path.getsize(MODEL_PATH)
        / (1024 * 1024),
        2
    ),
    "MB"
)

print(
    "Classes:",
    len(class_names)
)

print(
    "Class names:",
    list(class_names.values())
)

print("\n" + "=" * 70)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)