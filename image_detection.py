from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")

# Upload/select your image path
image_path = input("Enter the image path: ")

# Run detection
results = model.predict(
    source=image_path,
    conf=0.25,
    imgsz=640,
    save=True,
    show=True
)

print("\nDetection completed.")
print("Detected image saved in the 'runs/detect' folder.")