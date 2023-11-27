from ultralytics import YOLO

def predict():
    # Load a model
    # model = YOLO('yolov8n.pt')  # 加载官方的模型权重作评估
    model = YOLO('/home/xnwu/wangyong/Code/yolov8/runs/detect/yolov8s_train_dataset53/weights/best.pt')  # 加载自定义的模型权重作评估

   	# 评估
    imgsz=640
    metrics = model.val(imgsz=640)  # 不需要传参，这里定义的模型会自动在训练的数据集上作评估
    print("======>imsize:",imgsz)
    print(metrics.box.map)  # map50-95
    print(metrics.box.map50)  # map50
    print(metrics.box.map75)  # map75
    print(metrics.box.maps)  # 包含每个类别的map50-95列表

predict()