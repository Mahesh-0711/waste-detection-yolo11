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
# LOAD MODEL
# ============================================================

print("=" * 70)
print("YOLO11 OBJECT DETECTION ACCURACY / PERFORMANCE")
print("=" * 70)

model = YOLO(MODEL_PATH)

print("\nModel loaded successfully.")

# ============================================================
# VALIDATE MODEL
# ============================================================

print("\nRunning validation...\n")

metrics = model.val(
    data=DATA_YAML,
    split="val",
    imgsz=640,
    batch=16,
    device="cpu",
    plots=True,
    verbose=True
)

# ============================================================
# STANDARD OBJECT-DETECTION METRICS
# ============================================================

precision = float(metrics.box.mp)
recall = float(metrics.box.mr)
map50 = float(metrics.box.map50)
map5095 = float(metrics.box.map)

# F1 SCORE
if precision + recall > 0:
    f1 = (
        2 * precision * recall
        / (precision + recall)
    )
else:
    f1 = 0.0

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(f"Precision       : {precision:.4f}  ({precision * 100:.2f}%)")
print(f"Recall          : {recall:.4f}  ({recall * 100:.2f}%)")
print(f"mAP@50          : {map50:.4f}  ({map50 * 100:.2f}%)")
print(f"mAP@50-95       : {map5095:.4f}  ({map5095 * 100:.2f}%)")
print(f"F1 Score        : {f1:.4f}  ({f1 * 100:.2f}%)")

# ============================================================
# CLASS-WISE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CLASS-WISE RESULTS")
print("=" * 70)

for i, name in model.names.items():

    name = str(name)

    p = float(metrics.box.p[i])
    r = float(metrics.box.r[i])
    ap50 = float(metrics.box.ap50[i])
    ap = float(metrics.box.ap[i])

    if p + r > 0:
        class_f1 = (
            2 * p * r
            / (p + r)
        )
    else:
        class_f1 = 0.0

    print(
        f"\n{name}"
    )

    print(
        f"  Precision  : {p * 100:.2f}%"
    )

    print(
        f"  Recall     : {r * 100:.2f}%"
    )

    print(
        f"  mAP@50     : {ap50 * 100:.2f}%"
    )

    print(
        f"  mAP@50-95  : {ap * 100:.2f}%"
    )

    print(
        f"  F1 Score   : {class_f1 * 100:.2f}%"
    )

# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print("Model        : YOLO11n")
print("Classes      :", len(model.names))

print(
    "Model size   :",
    round(
        os.path.getsize(MODEL_PATH)
        / (1024 * 1024),
        2
    ),
    "MB"
)

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)