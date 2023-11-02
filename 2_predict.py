# Load YOLOv8n, train it on COCO128 for 3 epochs and predict an image with it
from ultralytics import YOLO

# model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model
# model.train(data='coco128.yaml', epochs=3)  # train the model
# model('https://ultralytics.com/images/bus.jpg')  # predict on an image
model = YOLO('/home/xnwu/wangyong/yolov8/runs/detect/train_on_dataset2/weights/best.pt')
model.predict('/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230517/20230517_for_det',imgsz=640,save=True,save_crop=True,device='cpu',vid_stride=10,show=False,conf=0.25)

# /home/xnwu/vims/数据采集/DVR/DVR_剪辑后回传/LOOP/no_elevated/crossroad/20221124142439077_LGWEF6A75MH250240_0_0_0.mp4
#文件夹下的图片：/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230428/20230428_for_det/1
# /media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230522/no_elevated/crosswalk
#            
#视频：/home/xnwu/vlc-record-2020-12-09-10h24m20s-rtsp___192.168.1.3_stream0-.mp4
   #/home/xnwu/vims/数据采集/DVR/20230718/20230718153307817_LGWEF6A75MH250240_0_0_0.mp4
   #'/home/xnwu/vims/数据采集/DVR/20230718/20230718153007805_LGWEF6A75MH250240_0_0_0.mp4'

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