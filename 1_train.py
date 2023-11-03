####训练出来看：效果？测试用数据增强了？ 后处理？ 是半精度运行？输入多大？网络多大？ 模型某些操作(5维)硬件支持？
# ###预打标只看效果 
from ultralytics import YOLO

 
# Ultralytics YOLO 🚀, GPL-3.0 license
'''
    项目一：路面标志检测
        csdn:https://blog.csdn.net/cg129054036/article/details/121109555
        数据集: https://github.com/oshadajay/CeyMo/blob/main/eval.py
        结果: Speed: 33.2ms preprocess, 660.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
            Speed: 12.2ms preprocess, 59.6ms inference, 0.7ms postprocess per image at shape (1, 3, 192, 320)
            Speed: 12.6ms preprocess, 46.6ms inference, 0.6ms postprocess per image at shape (1, 3, 160, 256)

        数据集：公司，见标注文档：
            
''' 
### 调节默认参数在 "/root/YOLOv8-main/ultralytics/yolo/configs/default.yaml"中
### 训练 yolo task=detect mode=train model=yolov8s.yaml  data=score_data.yaml epochs=100 batch=64 imgsz=640 pretrained=False optimizer=SGD 
### 预测 yolo task=detect mode=predict model=/root/YOLOv8-main/runs/detect/train9/weights/best.pt conf=0.25 source=/root/YOLOv8-main/rode_face_test/images    
# Load a model
model = YOLO('yolov8m.yaml')  # build a new model from YAML   
# model = YOLO('runs/detect/train_on_dataset1/weights/best.pt')  # load a pretrained model (recommended for training)#'
# model = model.load('/root/YOLOv8-main/ultralytics/yolo/v8/detect/yolov8n.pt')
model = model.load('/home/xnwu/wangyong/yolov8/official_weights/yolov8m.pt')
model.train(data='score_data.yaml', epochs=100, imgsz=640,batch=8)




'''
    1.报错 FileNotFoundError: /home/xnwu/wangyong/yolov8/ultralytics/assets/bus.jpg does not exist？
        放一张假图片bus.jpg进去。

'''
'''
    下列是可传入train参数：

    Key	   Value	Description
    model	None	模型路径. yolov8n.pt, yolov8n.yaml
    data	None	数据集路径, i.e. coco128.yaml
    epochs	100	    训练的总轮次
    patience	50	等待没有明显的改善，尽早停止训练的轮次
    batch	16	number of images per batch (-1 for AutoBatch)
    imgsz	640	size of input images as integer or w,h
    save	True	save train checkpoints and predict results
    save_period	-1	保存检查点每x个epoch ( (disabled if < 1)
    cache	False	True/ram, disk or False. 用于数据加载的Usle缓存
    device	None	device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
    workers	8	number of worker threads for data loading (per RANK if DDP) 用于数据加载的工作线程数
    project	None	project name 项目名称
    name	None	experiment name 实验名称
    exist_ok	False	whether to overwrite existing experiment 是否覆盖现有实验
    pretrained	False	whether to use a pretrained model 是否使用预训练的模型
    optimizer	'SGD'	optimizer to use, choices=[‘SGD’, ‘Adam’, ‘AdamW’, ‘RMSProp’] 使用优化器，
    verbose	False	whether to print verbose output 是否打印详细输出
    seed	0	random seed for reproducibility 随机种子的重现性
    deterministic	True	whether to enable deterministic mode 是否启用确定性模式
    single_cls	False	train multi-class data as single-class 将多类数据训练为单类
    image_weights	False	use weighted image selection for training 使用加权图像选择进行训练
    rect	False	rectangular training with each batch collated for minimum padding
    cos_lr	False	use cosine learning rate scheduler
    close_mosaic	0	(int) disable mosaic augmentation for final epochs
    resume	False	resume training from last checkpoint
    amp	True	Automatic Mixed Precision (AMP) training, choices=[True, False]
    lr0	0.01	initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    lrf	0.01	final learning rate (lr0 * lrf)
    momentum	0.937	SGD momentum/Adam beta1
    weight_decay	0.0005	optimizer weight decay 5e-4
    warmup_epochs	3.0	warmup epochs (fractions ok)
    warmup_momentum	0.8	warmup initial momentum
    warmup_bias_lr	0.1	warmup initial bias lr
    box	7.5	box loss gain
    cls	0.5	cls loss gain (scale with pixels)
    dfl	1.5	dfl loss gain
    pose	12.0	pose loss gain (pose-only)
    kobj	2.0	keypoint obj loss gain (pose-only)
    label_smoothing	0.0	label smoothing (fraction)
    nbs	64	nominal batch size
    overlap_mask	True	masks should overlap during training (segment train only)
    mask_ratio	4	mask downsample ratio (segment train only)
    dropout	0.0	use dropout regularization (classify train only)
    val	True	validate/test during training
    ————————————————
    版权声明：本文为CSDN博主「Deen..」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
    原文链接：https://blog.csdn.net/weixin_55224780/article/details/130154135
'''