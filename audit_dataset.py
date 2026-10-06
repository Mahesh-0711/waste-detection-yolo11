from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import hashlib
import random
import csv
import math

# ============================================================
# CONFIGURATION
# ============================================================

DATASET = Path(r"D:\waste_detection_project\content\waste_detection")

OUTPUT = DATASET.parent / "dataset_audit"

CLASS_NAMES = {
    0: "Glass",
    1: "Metal",
    2: "Paper",
    3: "Plastic",
    4: "Waste"
}

SPLITS = ["train", "valid", "test"]

# Number of examples to visually inspect for each important class
SAMPLES_PER_CLASS = 100

# Very small object threshold
SMALL_OBJECT_RATIO = 0.005

# ============================================================
# SETUP
# ============================================================

OUTPUT.mkdir(parents=True, exist_ok=True)

for split in SPLITS:
    (OUTPUT / split).mkdir(parents=True, exist_ok=True)

random.seed(42)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def image_files(folder):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    return [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ]


def md5_file(path):
    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def read_labels(label_path):
    """
    Reads YOLO labels:

    class x_center y_center width height
    """

    annotations = []
    errors = []

    if not label_path.exists():
        return annotations, ["missing_label"]

    try:
        text = label_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        return annotations, [f"read_error:{e}"]

    if not text:
        return annotations, ["empty_label"]

    for line_number, line in enumerate(text.splitlines(), 1):

        parts = line.strip().split()

        if len(parts) != 5:
            errors.append(
                f"line_{line_number}: expected_5_values"
            )
            continue

        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])

        except Exception:
            errors.append(
                f"line_{line_number}: invalid_number"
            )
            continue

        # Check class
        if cls not in CLASS_NAMES:
            errors.append(
                f"line_{line_number}: invalid_class_{cls}"
            )
            continue

        # Check coordinates
        values = [x, y, w, h]

        if any(not math.isfinite(v) for v in values):
            errors.append(
                f"line_{line_number}: non_finite_value"
            )
            continue

        if not (0 <= x <= 1):
            errors.append(
                f"line_{line_number}: x_out_of_range"
            )

        if not (0 <= y <= 1):
            errors.append(
                f"line_{line_number}: y_out_of_range"
            )

        if not (0 < w <= 1):
            errors.append(
                f"line_{line_number}: width_invalid"
            )

        if not (0 < h <= 1):
            errors.append(
                f"line_{line_number}: height_invalid"
            )

        # Bounding box boundaries
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
            errors.append(
                f"line_{line_number}: box_outside_image"
            )

        annotations.append({
            "class": cls,
            "x": x,
            "y": y,
            "w": w,
            "h": h
        })

    return annotations, errors


# ============================================================
# DRAW ANNOTATIONS
# ============================================================

def create_preview(image_path, label_path, output_path):

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return False

    W, H = img.size

    draw = ImageDraw.Draw(img)

    annotations, _ = read_labels(label_path)

    for ann in annotations:

        cls = ann["class"]

        x = ann["x"]
        y = ann["y"]
        w = ann["w"]
        h = ann["h"]

        x1 = int((x - w / 2) * W)
        y1 = int((y - h / 2) * H)

        x2 = int((x + w / 2) * W)
        y2 = int((y + h / 2) * H)

        name = CLASS_NAMES[cls]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3
        )

        draw.rectangle(
            [x1, max(0, y1 - 25), x1 + 100, y1],
            fill="red"
        )

        draw.text(
            (x1 + 3, max(0, y1 - 22)),
            name,
            fill="white"
        )

    img.thumbnail((1200, 900))

    img.save(output_path)

    return True


# ============================================================
# MAIN AUDIT
# ============================================================

print("=" * 70)
print("YOLO DATASET AUDIT")
print("=" * 70)

print(f"\nDataset:")
print(DATASET)

print(f"\nAudit output:")
print(OUTPUT)

if not DATASET.exists():
    print("\nERROR: Dataset folder does not exist.")
    print("Check the DATASET path at the top of this script.")
    raise SystemExit


# ============================================================
# STATISTICS
# ============================================================

class_image_count = {name: 0 for name in CLASS_NAMES.values()}
class_box_count = {name: 0 for name in CLASS_NAMES.values()}

split_image_count = {}
split_label_count = {}

small_objects = []
annotation_errors = []
missing_labels = []
empty_labels = []
duplicate_images = []

all_hashes = {}

class_files = {
    cls: []
    for cls in CLASS_NAMES
}

total_images = 0
total_boxes = 0
multi_object_images = 0
single_object_images = 0


# ============================================================
# SCAN DATASET
# ============================================================

for split in SPLITS:

    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    if not image_dir.exists():
        print(f"\nWARNING: Missing folder: {image_dir}")
        continue

    files = image_files(image_dir)

    split_image_count[split] = len(files)

    labels_found = 0

    print(f"\nScanning {split}: {len(files)} images")

    for image_path in files:

        total_images += 1

        # ----------------------------------------------------
        # Duplicate check
        # ----------------------------------------------------

        try:
            file_hash = md5_file(image_path)

            if file_hash in all_hashes:

                duplicate_images.append({
                    "image": str(image_path),
                    "duplicate_of": all_hashes[file_hash]
                })

            else:
                all_hashes[file_hash] = str(image_path)

        except Exception as e:
            annotation_errors.append({
                "image": str(image_path),
                "error": f"hash_error:{e}"
            })

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label_path = label_dir / f"{image_path.stem}.txt"

        annotations, errors = read_labels(label_path)

        if not label_path.exists():

            missing_labels.append(str(image_path))

        else:

            labels_found += 1

            if not annotations:
                empty_labels.append(str(image_path))

        # ----------------------------------------------------
        # Annotation errors
        # ----------------------------------------------------

        if errors:

            for error in errors:

                annotation_errors.append({
                    "image": str(image_path),
                    "error": error
                })

        # ----------------------------------------------------
        # Object counts
        # ----------------------------------------------------

        if len(annotations) > 1:
            multi_object_images += 1

        elif len(annotations) == 1:
            single_object_images += 1

        total_boxes += len(annotations)

        # ----------------------------------------------------
        # Image size
        # ----------------------------------------------------

        try:

            with Image.open(image_path) as im:
                W, H = im.size

        except Exception as e:

            annotation_errors.append({
                "image": str(image_path),
                "error": f"image_open_error:{e}"
            })

            continue

        # ----------------------------------------------------
        # Process annotations
        # ----------------------------------------------------

        for ann in annotations:

            cls = ann["class"]

            class_name = CLASS_NAMES[cls]

            class_box_count[class_name] += 1

            class_image_count[class_name] += 1

            class_files[cls].append(
                (image_path, label_path)
            )

            # Object area
            area_ratio = ann["w"] * ann["h"]

            if area_ratio < SMALL_OBJECT_RATIO:

                small_objects.append({
                    "image": str(image_path),
                    "class": class_name,
                    "area_ratio": area_ratio
                })

    split_label_count[split] = labels_found


# ============================================================
# REPORT
# ============================================================

print("\n")
print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"Total images: {total_images}")
print(f"Total bounding boxes: {total_boxes}")

print("\nImages by split:")

for split in SPLITS:

    print(
        f"  {split:8s}: "
        f"{split_image_count.get(split, 0)} images | "
        f"{split_label_count.get(split, 0)} labels"
    )

print("\nObjects per class:")

for cls, name in CLASS_NAMES.items():

    print(
        f"  {name:8s}: "
        f"{class_box_count[name]} boxes"
    )

print("\nImages containing each class:")

for cls, name in CLASS_NAMES.items():

    # Remove duplicates because one image may contain multiple objects
    unique_images = set(
        str(x[0])
        for x in class_files[cls]
    )

    class_image_count[name] = len(unique_images)

    print(
        f"  {name:8s}: "
        f"{class_image_count[name]} images"
    )

print("\nImage type:")

print(f"  Single-object images : {single_object_images}")
print(f"  Multi-object images  : {multi_object_images}")

print("\nPotential problems:")

print(f"  Missing labels       : {len(missing_labels)}")
print(f"  Empty labels         : {len(empty_labels)}")
print(f"  Annotation errors    : {len(annotation_errors)}")
print(f"  Duplicate images     : {len(duplicate_images)}")
print(f"  Small-object boxes   : {len(small_objects)}")


# ============================================================
# SAVE CSV REPORTS
# ============================================================

def save_csv(filename, rows, fieldnames):

    path = OUTPUT / filename

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


save_csv(
    "missing_labels.csv",
    [{"image": x} for x in missing_labels],
    ["image"]
)

save_csv(
    "empty_labels.csv",
    [{"image": x} for x in empty_labels],
    ["image"]
)

save_csv(
    "annotation_errors.csv",
    annotation_errors,
    ["image", "error"]
)

save_csv(
    "duplicate_images.csv",
    duplicate_images,
    ["image", "duplicate_of"]
)

save_csv(
    "small_objects.csv",
    small_objects,
    ["image", "class", "area_ratio"]
)


# ============================================================
# CREATE VISUAL REVIEW SAMPLES
# ============================================================

print("\n")
print("=" * 70)
print("CREATING VISUAL REVIEW SAMPLES")
print("=" * 70)

for cls in [3, 4]:

    class_name = CLASS_NAMES[cls]

    files = class_files[cls]

    if not files:
        print(f"No images found for {class_name}")
        continue

    random.shuffle(files)

    selected = files[:SAMPLES_PER_CLASS]

    output_dir = OUTPUT / class_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nCreating {len(selected)} "
        f"{class_name} review images..."
    )

    for index, (image_path, label_path) in enumerate(selected, 1):

        output_file = (
            output_dir /
            f"{index:03d}_{image_path.name}"
        )

        create_preview(
            image_path,
            label_path,
            output_file
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print(f"""
Audit folder:

{OUTPUT}

Important folders:

{OUTPUT / "Plastic"}
{OUTPUT / "Waste"}

Open these two folders and manually inspect the bounding boxes.

Check especially:

1. Is the object actually Plastic/Waste?
2. Is the bounding box tight?
3. Is the entire object covered?
4. Is unnecessary background included?
5. Is there an object that should have been annotated but wasn't?
6. Is a Plastic object incorrectly labelled Waste?
7. Is a Waste object incorrectly labelled Plastic?

Do NOT delete or modify the original dataset yet.
""")

print("=" * 70)