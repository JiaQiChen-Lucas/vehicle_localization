from ultralytics import YOLO
from ultralytics.nn.Addmodules import C2REPA, REPAPoseHead

'''
model = YOLO(model="/media/hardDisk1/chenjiaqi/code/vehicle_yolov8.2.15/ultralytics-8.2.15/result2/baseline/yolov8n-pose-ReLU/train3/weights/best_pck_oks.pt")
model.export(format='onnx', opset=12)
'''

model = YOLO(model="/media/hardDisk1/lucas/code/Wheel-YOLOv8-pose/ultralytics-8.2.15/result/Wheel-YOLOv8-pose/ReLU/train/weights/best_pck_oks.pt")
print(model.info())

for name, module in model.named_modules():
    if isinstance(module, C2REPA) or isinstance(module, REPAPoseHead):
        print(f"Switching {name} to deploy mode...")
        module.switch_to_deploy()

model.export(format='onnx', opset=12)
