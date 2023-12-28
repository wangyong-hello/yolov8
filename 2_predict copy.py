from pickle import FALSE
from ultralytics import YOLO
import os,cv2
from tqdm import tqdm

# # note:推理视频看效果前，要删除挑选图片模块
model = YOLO('/home/xnwu/wangyong/Code/Yolov8/runs/detect/yolov8n_train_dataset9_norect_no_fliplr_no_scale_crop_val2/weights/best.pt') 
model.predict("/home/xnwu/wangyong/vims/数据采集/DVR/行车记录仪dingdingpai/20201221/20201221223743_0060.mp4",imgsz=320,save=False,save_crop=False,device='cuda',vid_stride=15,show=True,conf=0.3)
