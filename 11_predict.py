# Load YOLOv8n, train it on COCO128 for 3 epochs and predict an image with it
from ultralytics import YOLO

# model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model
# model.train(data='coco128.yaml', epochs=3)  # train the model
# model('https://ultralytics.com/images/bus.jpg')  # predict on an image
model = YOLO(r'C:\Users\wangyong\Desktop\YOLOv8-main\runs\detect\train9\weights\best.pt')
model(r'C:\Users\wangyong\Desktop\YOLOv8-main\20230713121305931_LGWEF6A75MH250240_0_0_0.mp4',save=True,save_crop=True,vid_stride=10)