from pathlib import Path
from PIL import Image, ImageDraw
import hashlib
import shutil
import random
import csv
import math

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE = Path(
    r"D:\waste_detection_project\content\waste_detection"
)

DEST = Path(
    r"D:\waste_detection_project\content\waste_detection_clean"
)

CLASS_NAMES = {
    0: "Glass",
    1: "Metal",
    2: "Paper",
    3: "Plastic",
    4: "Waste"
}

SPLITS = ["train", "valid", "test"]

# Number of Plastic/Waste images for visual review
REVIEW_SAMPLES = 100

# ============================================================
# SAFETY
# ============================================================

if not SOURCE.exists():
    print("ERROR: Source dataset does not exist:")
    print(SOURCE)
    raise SystemExit(1)

if DEST.exists():
    print("ERROR: Destination already exists:")
    print(DEST)
    print()
    print("This script will NOT overwrite an existing cleaned dataset.")
    print("Delete or rename that folder first if you want to run again.")
    raise SystemExit(1)

DEST.mkdir(parents=True)

print("=" * 70)
print("YOLO DATASET CLEANING")
print("=" * 70)

print("\nSOURCE:")
print(SOURCE)

print("\nDESTINATION:")
print(DEST)

print("\nOriginal dataset will NOT be modified.")


# ============================================================
# REPORT
# ============================================================

report = {
    "images_copied": 0,
    "images_removed_duplicate": 0,
    "boxes_processed": 0,
    "boxes_clipped": 0,
    "invalid_lines_removed": 0,
    "duplicate_boxes_removed": 0,
    "empty_labels": 0,
    "missing_labels": 0,
    "corrupt_images": 0,
}


# ============================================================
# HELPERS
# ============================================================

def image_files(folder):

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    if not folder.exists():
        return []

    return sorted([
        f for f in folder.iterdir()
        if f.is_file()
        and f.suffix.lower() in extensions
    ])


def md5_file(path):

    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def parse_label_file(label_path):

    """
    Read YOLO labels and clean:

    class x_center y_center width height

    Invalid lines are removed.

    Bounding boxes are clipped to [0,1].
    """

    boxes = []

    if not label_path.exists():

        return boxes, {
            "missing": True,
            "empty": False
        }

    try:

        text = label_path.read_text(
            encoding="utf-8"
        ).strip()

    except Exception:

        return boxes, {
            "missing": False,
            "empty": True
        }

    if not text:

        return boxes, {
            "missing": False,
            "empty": True
        }

    for line_number, line in enumerate(
        text.splitlines(),
        1
    ):

        parts = line.split()

        if len(parts) != 5:

            report["invalid_lines_removed"] += 1

            continue

        try:

            cls = int(float(parts[0]))

            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

        except Exception:

            report["invalid_lines_removed"] += 1

            continue

        if cls not in CLASS_NAMES:

            report["invalid_lines_removed"] += 1

            continue

        values = [x, y, w, h]

        if any(
            not math.isfinite(v)
            for v in values
        ):

            report["invalid_lines_removed"] += 1

            continue

        if w <= 0 or h <= 0:

            report["invalid_lines_removed"] += 1

            continue

        # ----------------------------------------------------
        # Original box
        # ----------------------------------------------------

        x1 = x - w / 2
        y1 = y - h / 2

        x2 = x + w / 2
        y2 = y + h / 2

        original = (
            x1,
            y1,
            x2,
            y2
        )

        # ----------------------------------------------------
        # Clip to image
        # ----------------------------------------------------

        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))

        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))

        if original != (
            x1,
            y1,
            x2,
            y2
        ):

            report["boxes_clipped"] += 1

        new_w = x2 - x1
        new_h = y2 - y1

        if new_w <= 0 or new_h <= 0:

            report["invalid_lines_removed"] += 1

            continue

        new_x = (x1 + x2) / 2
        new_y = (y1 + y2) / 2

        box = (
            cls,
            round(new_x, 6),
            round(new_y, 6),
            round(new_w, 6),
            round(new_h, 6)
        )

        # ----------------------------------------------------
        # Remove duplicate boxes
        # ----------------------------------------------------

        if box in boxes:

            report["duplicate_boxes_removed"] += 1

            continue

        boxes.append(box)

        report["boxes_processed"] += 1

    return boxes, {
        "missing": False,
        "empty": False
    }


def save_label_file(path, boxes):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for cls, x, y, w, h in boxes:

            f.write(
                f"{cls} {x} {y} {w} {h}\n"
            )


def draw_boxes(
    image_path,
    label_path,
    output_path
):

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

    except Exception:

        return False

    W, H = image.size

    draw = ImageDraw.Draw(image)

    boxes, status = parse_label_file(
        label_path
    )

    for cls, x, y, w, h in boxes:

        x1 = int(
            (x - w / 2) * W
        )

        y1 = int(
            (y - h / 2) * H
        )

        x2 = int(
            (x + w / 2) * W
        )

        y2 = int(
            (y + h / 2) * H
        )

        name = CLASS_NAMES[cls]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3
        )

        draw.text(
            (x1 + 3, max(0, y1 + 3)),
            name,
            fill="red"
        )

    image.thumbnail(
        (1200, 900)
    )

    image.save(
        output_path
    )

    return True


# ============================================================
# STEP 1
# FIND DUPLICATES
# ============================================================

print("\n")
print("=" * 70)
print("STEP 1 — FINDING DUPLICATES")
print("=" * 70)

hash_map = {}

for split in SPLITS:

    image_dir = (
        SOURCE /
        split /
        "images"
    )

    for image in image_files(
        image_dir
    ):

        try:

            h = md5_file(image)

        except Exception:

            continue

        if h not in hash_map:

            hash_map[h] = []

        hash_map[h].append(
            (split, image)
        )


# ============================================================
# DETERMINE FILES TO REMOVE
# ============================================================

remove_files = set()

duplicate_report = []

for h, items in hash_map.items():

    if len(items) <= 1:
        continue

    # --------------------------------------------------------
    # Priority:
    # test > valid > train
    # --------------------------------------------------------

    priority = {
        "test": 0,
        "valid": 1,
        "train": 2
    }

    items_sorted = sorted(
        items,
        key=lambda x: priority[x[0]]
    )

    keep_split, keep_file = items_sorted[0]

    for split, file in items_sorted[1:]:

        # If same split, keep only one
        remove_files.add(
            str(file.resolve())
        )

        duplicate_report.append({
            "removed": str(file),
            "kept": str(keep_file),
            "removed_split": split,
            "kept_split": keep_split
        })

print(
    f"Duplicate files marked for removal: "
    f"{len(remove_files)}"
)

report[
    "images_removed_duplicate"
] = len(remove_files)


# ============================================================
# STEP 2
# CREATE DATASET
# ============================================================

print("\n")
print("=" * 70)
print("STEP 2 — CREATING CLEAN DATASET")
print("=" * 70)

empty_review = []

for split in SPLITS:

    source_images = (
        SOURCE /
        split /
        "images"
    )

    source_labels = (
        SOURCE /
        split /
        "labels"
    )

    dest_images = (
        DEST /
        split /
        "images"
    )

    dest_labels = (
        DEST /
        split /
        "labels"
    )

    dest_images.mkdir(
        parents=True,
        exist_ok=True
    )

    dest_labels.mkdir(
        parents=True,
        exist_ok=True
    )

    images = image_files(
        source_images
    )

    print(
        f"\nProcessing {split}: "
        f"{len(images)} images"
    )

    for image_path in images:

        # ----------------------------------------------------
        # Skip duplicate
        # ----------------------------------------------------

        if str(
            image_path.resolve()
        ) in remove_files:

            continue

        # ----------------------------------------------------
        # Verify image
        # ----------------------------------------------------

        try:

            with Image.open(
                image_path
            ) as img:

                img.verify()

        except Exception:

            report[
                "corrupt_images"
            ] += 1

            print(
                "Corrupt image:",
                image_path
            )

            continue

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label_path = (
            source_labels /
            f"{image_path.stem}.txt"
        )

        boxes, status = parse_label_file(
            label_path
        )

        if status["missing"]:

            report[
                "missing_labels"
            ] += 1

        if status["empty"]:

            report[
                "empty_labels"
            ] += 1

            empty_review.append(
                str(image_path)
            )

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        shutil.copy2(
            image_path,
            dest_images /
            image_path.name
        )

        # ----------------------------------------------------
        # Save cleaned labels
        # ----------------------------------------------------

        save_label_file(
            dest_labels /
            f"{image_path.stem}.txt",
            boxes
        )

        report[
            "images_copied"
        ] += 1


# ============================================================
# STEP 3
# SAVE DUPLICATE REPORT
# ============================================================

if duplicate_report:

    with open(
        DEST /
        "removed_duplicates.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "removed",
                "kept",
                "removed_split",
                "kept_split"
            ]
        )

        writer.writeheader()

        writer.writerows(
            duplicate_report
        )


# ============================================================
# STEP 4
# SAVE EMPTY LABEL REVIEW
# ============================================================

review_dir = (
    DEST /
    "REVIEW_EMPTY_LABELS"
)

review_dir.mkdir(
    parents=True,
    exist_ok=True
)

for image_string in empty_review:

    image_path = Path(
        image_string
    )

    label_path = (
        SOURCE /
        image_path.parent.parent.name /
        "labels" /
        f"{image_path.stem}.txt"
    )

    output_path = (
        review_dir /
        image_path.name
    )

    shutil.copy2(
        image_path,
        output_path
    )


# ============================================================
# STEP 5
# CREATE PLASTIC / WASTE REVIEW
# ============================================================

print("\n")
print("=" * 70)
print("STEP 3 — CREATING PLASTIC/WASTE REVIEW")
print("=" * 70)

review_base = (
    DEST /
    "REVIEW_CLASSES"
)

for class_id in [3, 4]:

    class_name = CLASS_NAMES[class_id]

    output_dir = (
        review_base /
        class_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    candidates = []

    for split in SPLITS:

        image_dir = (
            DEST /
            split /
            "images"
        )

        label_dir = (
            DEST /
            split /
            "labels"
        )

        for image in image_files(
            image_dir
        ):

            label = (
                label_dir /
                f"{image.stem}.txt"
            )

            boxes, _ = parse_label_file(
                label
            )

            contains_class = any(
                b[0] == class_id
                for b in boxes
            )

            if contains_class:

                candidates.append(
                    (image, label)
                )

    random.shuffle(
        candidates
    )

    selected = candidates[
        :REVIEW_SAMPLES
    ]

    print(
        f"{class_name}: "
        f"{len(selected)} review images"
    )

    for i, (
        image,
        label
    ) in enumerate(
        selected,
        1
    ):

        output = (
            output_dir /
            f"{i:03d}_{image.name}"
        )

        draw_boxes(
            image,
            label,
            output
        )


# ============================================================
# STEP 6
# CREATE DATA.YAML
# ============================================================

yaml_content = f"""path: {DEST.resolve().as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 5
names:
  - Glass
  - Metal
  - Paper
  - Plastic
  - Waste
"""

(
    DEST /
    "data.yaml"
).write_text(
    yaml_content,
    encoding="utf-8"
)


# ============================================================
# STEP 7
# SAVE REPORT
# ============================================================

report_file = (
    DEST /
    "CLEANING_REPORT.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "YOLO DATASET CLEANING REPORT\n"
    )

    f.write(
        "=" * 60 +
        "\n\n"
    )

    for key, value in report.items():

        f.write(
            f"{key}: {value}\n"
        )

    f.write("\n")

    f.write(
        "IMPORTANT:\n"
    )

    f.write(
        "Small objects were NOT removed automatically.\n"
    )

    f.write(
        "Empty-label images were preserved for manual review.\n"
    )

    f.write(
        "Plastic/Waste labels were NOT automatically changed.\n"
    )

    f.write(
        "Semantic class correctness requires visual inspection.\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)

print("\nClean dataset:")
print(DEST)

print("\nReport:")
print(report_file)

print("\nImportant review folders:")

print(
    DEST /
    "REVIEW_EMPTY_LABELS"
)

print(
    DEST /
    "REVIEW_CLASSES" /
    "Plastic"
)

print(
    DEST /
    "REVIEW_CLASSES" /
    "Waste"
)

print("\nSummary:")

for key, value in report.items():

    print(
        f"{key:30s}: {value}"
    )

print("\n")
print("IMPORTANT:")
print(
    "The original dataset was NOT modified."
)

print(
    "\nBefore retraining, inspect:"
)

print(
    "1. REVIEW_EMPTY_LABELS"
)

print(
    "2. REVIEW_CLASSES\\Plastic"
)

print(
    "3. REVIEW_CLASSES\\Waste"
)

print(
    "\nIf these look correct, use:"
)

print(
    DEST /
    "data.yaml"
)

print(
    "for YOLO11 retraining."
)