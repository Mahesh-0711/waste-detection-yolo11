from pathlib import Path
from collections import Counter
import yaml

DATASET = Path(r"D:\waste_detection_project\content\waste_detection_clean")

CLASS_NAMES = ["Glass", "Metal", "Paper", "Plastic", "Waste"]

splits = ["train", "valid", "test"]

print("=" * 60)
print("CLEAN DATASET VERIFICATION")
print("=" * 60)

total_images = 0
total_boxes = 0
class_counts = Counter()

for split in splits:
    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    images = list(image_dir.glob("*"))
    labels = list(label_dir.glob("*.txt"))

    image_stems = {x.stem for x in images}
    label_stems = {x.stem for x in labels}

    missing_labels = image_stems - label_stems
    extra_labels = label_stems - image_stems

    print(f"\n{split.upper()}")
    print("-" * 40)
    print("Images :", len(images))
    print("Labels :", len(labels))
    print("Missing labels:", len(missing_labels))
    print("Extra labels  :", len(extra_labels))

    total_images += len(images)

    for label_file in labels:
        try:
            lines = label_file.read_text().splitlines()

            for line in lines:
                parts = line.split()

                if len(parts) != 5:
                    print("BAD FORMAT:", label_file)
                    continue

                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:])

                if cls < 0 or cls >= len(CLASS_NAMES):
                    print("BAD CLASS:", label_file)
                    continue

                class_counts[CLASS_NAMES[cls]] += 1
                total_boxes += 1

        except Exception as e:
            print("ERROR:", label_file, e)

print("\n" + "=" * 60)
print("TOTAL")
print("=" * 60)

print("Total images:", total_images)
print("Total boxes :", total_boxes)

print("\nObjects per class:")

for cls in CLASS_NAMES:
    print(f"{cls:10s}: {class_counts[cls]}")

print("\nClass distribution:")
total = sum(class_counts.values())

for cls in CLASS_NAMES:
    percentage = (
        class_counts[cls] / total * 100
        if total > 0 else 0
    )
    print(f"{cls:10s}: {percentage:.2f}%")

print("\nVerification complete.")