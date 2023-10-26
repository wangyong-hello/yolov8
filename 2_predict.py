# Load YOLOv8n, train it on COCO128 for 3 epochs and predict an image with it
from ultralytics import YOLO

# model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model
# model.train(data='coco128.yaml', epochs=3)  # train the model
# model('https://ultralytics.com/images/bus.jpg')  # predict on an image
model = YOLO('/home/xnwu/wangyong/yolov8/runs/detect/train7/weights/best.pt')
model.predict('/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230522/no_elevated/crosswalk',save=True,save_crop=True,vid_stride=10)


    ####训练出来看：效果？测试用数据增强了？ 后处理？ 是半精度运行？输入多大？网络多大？ 模型某些操作(5维)硬件支持？
    # ###预打标只看效果 
    # from ultralytics import YOLO
        
    # # Load a model
    # model = YOLO(r'C:\Users\wangyong\Desktop\YOLOv8-main\ultralytics\yolo\v8\models\yolov8n.yaml')  # build a new model from YAML   
    # model = YOLO(r'C:\Users\wangyong\Desktop\YOLOv8-main\runs\detect\train9\weights\best.pt')  # load a pretrained model (recommended for training)
    # # model = model.load('/root/YOLOv8-main/ultralytics/yolo/v8/detect/yolov8n.pt')
    # model.predict(source=r'C:\Users\wangyong\Desktop\YOLOv8-main\20230713121305931_LGWEF6A75MH250240_0_0_0.mp4',save_crop=True,vid_stride=10)

