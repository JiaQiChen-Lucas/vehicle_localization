from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v8/Wheel-YOLOv8-pose.yaml", task="pose")
model.train(task="pose", data="./ultralytics/cfg/datasets/engineering_vehicle.yaml", epochs=800, batch=64, workers=8, device=0, lr0=0.001, project="result/Wheel-YOLOv8-pose", pretrained=False, seed=42, save_period=1, patience=0, close_mosaic=0)
