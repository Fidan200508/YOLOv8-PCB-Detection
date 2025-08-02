from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")  # Load your trained model

results = model(r"C:\Users\Fidan\PycharmProjects\DetectingComponentsOnPCB\test2_pcb.jpg")  # Use your full image path

results[0].show()  # Display image with detections (if any)
