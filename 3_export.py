#CLI
# yolo task=detect mode=export model=./runs/detect/train/weights/last.pt format=onnx simplify=True opset=13

# python
from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/last.pt ")  # load a pretrained YOLOv8n model
model.export(format="onnx")  # export the model to ONNX format