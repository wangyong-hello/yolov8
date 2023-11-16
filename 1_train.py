####训练出来看：效果？测试用数据增强了？ 后处理？ 是半精度运行？输入多大？网络多大？ 模型某些操作(5维)硬件支持？
# ###预打标只看效果 
from ultralytics import YOLO

# from tensorboardX import summary
 
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

## note:仅放入权重也可以构建网络并导入权重，不需要先构建网络再导入权重后进行预训练
# model = YOLO('/home/xnwu/wangyong/yolov8/runs/detect/train/weights/best.pt')  # build a new model from YAML   
# # model = YOLO('runs/detect/train_on_dataset1/weights/best.pt')  # load a pretrained model (recommended for training)#'
# # model = model.load('/home/xnwu/wangyong/yolov8/runs/detect/train_on_dataset2/weights/best.pt')
# 

# model=YOLO('yolov8_p2_cbam.yaml').load('/home/xnwu/wangyong/yolov8/runs/detect/train_on_dataset3/weights/best.pt')
model=YOLO('yolov8m.yaml').load('/home/xnwu/wangyong/yolov8/runs/detect/train_yolov8m_on_dataset3/weights/best.pt')
model.train(data='ultralytics/cfg/score_data.yaml', epochs=100, imgsz=640,batch=8)

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
'''
    1.yolov8项目的文件结构：
        (1)ultralytics/cfg：
            主要是模型的配置文件(如：yolov8s.yaml)，超参数的配置文件,数据集的配置文件（类别、训练路径等）
        (2)ultralytics/nn：
            模型的conv、head、block基础构件在 ultralytics/nn/modules 中
            最后在task中注册模块并搭建网络
            
        (3)ultralytics/engine:
            模型实际的trainer，predictor，validator。定义了train、predict、val的整个过程
        (4)ultralytics/models：
            “有点像构建并声明” ultralytics/models/yolo/model.py----->   validator，predictor，trainer
        
            调用封装后的 ultralytics/models/yolo/detect/train.py-----> 'trainer': yolo.detect.DetectionTrainer,
                        ultralytics/models/yolo/detect/val.py----->'validator': yolo.detect.DetectionValidator,
                        ultralytics/models/yolo/detect/predict.py----->'predictor': yolo.detect.DetectionPredictor
                        
        (5)ultralytics/data：数据集的加载，增强，封包等
        (6)ultralytics/utils：画图、下载权重、等一些脚本
    
    2.添加注意力的流程：
        (1)在modules中指定文件中添加要新建的模块（注意力、特殊卷积模块也好）。如果构建了个新模块，就加在block.py中,如果是什么
        注意力，一般加在conv.py中。最后加完记得在文件头的__all___中写入，方便在task中导入
        (2)在task.py中导入并在prase_model()函数中注册加入模块，
        (3)最后在网络的yaml文件中进行配置
        
            
    2.报错 FileNotFoundError: /home/xnwu/wangyong/yolov8/ultralytics/assets/bus.jpg does not exist？
        放一张假图片bus.jpg进去。
    3.model = YOLO('yolov5s.yaml') ,使用本项目构建yolov5s网络，不是anchaor based。框和类别分离
      Anchor-free Split Ultralytics Head，见 https://docs.ultralytics.com/models/yolov5/#overview
    4.增加小目标检测层？
        https://github.com/ultralytics/ultralytics/issues/981
        YOLOv8-p2和YOLOv8-p6是YOLOv8目标检测模型的不同版本。

        YOLOv8—p2是YOLOv8的一个改进版本，它在原始的YOLOv8模型中新增了一个P2层。P2层做的卷积次数较少，特征图的尺寸较大，更适合用于小目标的识别。因此，YOLOv8—p2可以提升对小目标的检测能力。

        而YOLOv8—p6则是为了处理高分辨率图片而设计的一个版本。它在YOLOv8模型的基础上多卷积了一层，引入了更多的参数量。这使得YOLOv8—p6适用于处理高分辨率的图片，其中包含了大量可挖掘的信息。

        所以，YOLOv8—p2和YOLOv8—p6都是对YOLOv8模型的扩展和改进，分别用于小目标检测和高分辨率图片处理。

'''
