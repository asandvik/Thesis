from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# look into pretrained data

results = model.train(data="data.yaml", batch=0.8, patience=20, device=0, workers=4)