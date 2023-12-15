#CLI
# yolo task=detect mode=export model=./runs/detect/train/weights/last.pt format=onnx simplify=True opset=13

# python
from ultralytics import YOLO

model = YOLO("/home/xnwu/wangyong/Code/Yolov8/runs/detect/yolov8n_train_dataset8_new_norect_no_fliplr_no_scale/weights/best.pt")  # load a pretrained YOLOv8n model
model.export(format="onnx",imgsz=[192,320])  # export the model to ONNX format


# for i,name in enumerate(['LA', 'PC', 'SA', 'RA', 'SLA', 'SRA', 'MAR', 'CL', 'STA', 'NTA', 'LRA', 'MAL', 'SLR', 'other', 'TA', 'LTA']):
#     # print(f"{i}: {name}")
#     print(name)