# Load YOLOv8n, train it on COCO128 for 3 epochs and predict an image with it
from pickle import FALSE
from ultralytics import YOLO
import os,cv2
from tqdm import tqdm

# # note:推理视频看效果前，要删除挑选图片模块
model = YOLO('/home/xnwu/wangyong/Code/Yolov8/runs/detect/yolov8n_train_dataset8_no_fliplr_no_scale_no_rect_no_translate/weights/best.pt') 
model.predict("/home/xnwu/wangyong/vims/20231122浦东-中环-华夏-龙东-内环/场景理解_视频/高架下/20231122140603596_LGWEF6A75MH250240_0_0_0.mp4_20231124_141252.mp4",imgsz=320,save=False,save_crop=False,device='cuda',vid_stride=20,show=True,conf=0.3)
# /home/xnwu/wangyong/vims/20231122浦东-中环-华夏-龙东-内环/场景理解_视频/高架下/20231122140603596_LGWEF6A75MH250240_0_0_0.mp4_20231124_141252.mp4
# 20231122120936821_LGWEF6A75MH250240_0_0_0.mp4_20231124_140946.mp4  强光
# 20231122140904549_LGWEF6A75MH250240_0_0_0.mp4_20231124_141418.mp4

# model = YOLO('/home/xnwu/wangyong/Code/yolov8/runs/detect/yolov8s_train_dataset6/weights/best.pt')
# video_set_root='/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230823'
# video_set_root='/home/xnwu/wangyong/vims/20231107/场景理解_视频/晚上/'
# for video in tqdm(os.listdir(video_set_root)[:]):
#     video_path=os.path.join(video_set_root,video)
#     model.predict(video_path,imgsz=640,save=False,save_crop=True,device='cuda',vid_stride=20,show=False,conf=0.3)

#文件夹下的图片：/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230428/20230428_for_det/1
           
#视频：/home/xnwu/wangyong/Dataset/test/20230718153007805_LGWEF6A75MH250240_0_0_0.mp4
#视频：/home/xnwu/wangyong/Dataset/test/20230718153307817_LGWEF6A75MH250240_0_0_0.mp4
# /home/xnwu/wangyong/Dataset/test/20230823145154202_LGWEF6A75MH250240_0_0_0.mp4

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
    
    #1.yoloV8的输入和输出
        输入一张320*320的图片,输出三张特征图:第一个16为类别数
        (batch,(16+16*4),10,10),(batch,(16+16*4),20,20),(batch,(16+16*4),40,40)
        分别对应原图上的大中小目标,分别缩放32倍,16倍,8倍。
        decoupled head,网络会分成回归分支和分类分支，最后再汇总在一起，
        得到输出shape为 batch*80*2100。
        
        而yolov5输出三张特征图为
        (batch,3,10,10,21),(batch,3,20,20,21),(batch,3,40,40,21) 
        3表示3个anchor,21=16+4+1,分别为16个类概率,4为box,1为cls
        得到输出shape为batch*63*2100.
        
        
        参阅：https://zhuanlan.zhihu.com/p/665949863
    
    
'''