#CLI
# yolo task=detect mode=export model=./runs/detect/train/weights/last.pt format=onnx simplify=True opset=13

# python
from ultralytics import YOLO

# model = YOLO("/home/xnwu/wangyong/yolov8/runs/detect/yangfan_train_on_dataset2/weights/best.pt")  # load a pretrained YOLOv8n model
# model.export(format="onnx")  # export the model to ONNX format


# for i,name in enumerate(['LA', 'PC', 'SA', 'RA', 'SLA', 'SRA', 'MAR', 'CL', 'STA', 'NTA', 'LRA', 'MAL', 'SLR', 'other', 'TA', 'LTA']):
#     # print(f"{i}: {name}")
#     print(name)