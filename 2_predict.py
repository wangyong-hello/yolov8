# Load YOLOv8n, train it on COCO128 for 3 epochs and predict an image with it
from ultralytics import YOLO
import os

#note:推理视频看效果前，要删除挑选图片模块
model = YOLO('/home/xnwu/wangyong/code/yolov8/runs/detect/train_yolov8m_on_dataset3/weights/best.pt') 
# model.predict('/home/xnwu/wangyong/Dataset/test/20230823145154202_LGWEF6A75MH250240_0_0_0.mp4',imgsz=640,save=False,save_crop=False,device='cpu',vid_stride=10,show=True,conf=0.3)


video_set_root='/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230621_det'
for video in os.listdir(video_set_root):
    video_path=os.path.join(video_set_root,video)
    model.predict(video_path,imgsz=640,save=False,save_crop=True,device='cpu',vid_stride=15,show=False,conf=0.3)

#文件夹下的图片：/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230428/20230428_for_det/1
           
#视频：/home/xnwu/wangyong/code/20230718153007805_LGWEF6A75MH250240_0_0_0.mp4
#视频：/home/xnwu/wangyong/code/20230718153307817_LGWEF6A75MH250240_0_0_0.mp4

'''
        source	'ultralytics/assets'	图片或视频的源目录
        conf	0.25	用于检测的对象置信度阈值
        classes	None	按类过滤结果，即class=0，或class=[0,2,3]
        iou	0.7	NMS 的联合交集 (IoU) 阈值
       
        show	False	尽可能显示结果
        save	False	保存图像和结果
        save_crop	False	保存裁剪后的图像和结果
        save_txt	False	将结果保存为 .txt 文件
        
        half	False	使用半精度 (FP16)
        device	None	要运行的设备，即 cuda device=0/1/2/3 或 device=cpu
        vid_stride	False	视频帧率步幅
        
     
        save_conf	False	保存带有置信度分数的结果
        hide_labels	False	隐藏标签
        hide_conf	False	隐藏置信度分数
        
        visualize	False	可视化模型特征
        augment	False	将图像增强应用于预测源
        agnostic_nms	False	类别不可知的 NMS
        retina_masks	False	使用高分辨率分割蒙版
        boxes	True	在分割预测中显示框
        ————————————————
        版权声明：本文为CSDN博主「芒果汁没有芒果」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
        原文链接：https://blog.csdn.net/qq_38668236/article/details/128914899
'''


'''
    目标框在ultralytics/models/yolo/detect/predict.py中修改
    绘制保存/ultralytics/engine/results.py中修改
    保存crop的图片在ultralytics/utils/plotting.py中修改
'''