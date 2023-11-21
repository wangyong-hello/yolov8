# Ultralytics YOLO 🚀, AGPL-3.0 license

from genericpath import exists
from importlib.resources import path
from os import lseek
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops

import shutil,torch,os

def mkdir(dir):
    if  not os.path.exists(dir):
        os.mkdir(dir)

class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))  
            # #tag:修改 添加下面四行代码
            # if results[0].boxes.shape[0] == 0 :
            #     des_dir='/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230622/20230622_noObj'
            #     mkdir(des_dir)
            #     shutil.move(img_path,des_dir )
            
            # try:
            #     if torch.any(results[0].boxes.cls==0.) :  #  判断一个数是否在PyTorch张量中,torch.any(tensor == number)
            #         des_dir='/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230622/20230622_LA'
            #         mkdir(des_dir)  
            #         shutil.move(img_path,des_dir)
            # except:
            #     pass

            # try:
            #     if torch.any(results[0].boxes.cls==3.) :  #  判断一个数是否在PyTorch张量中,torch.any(tensor == number)
            #         des_dir='/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230622/20230622_RA'
            #         mkdir(des_dir)
            #         shutil.move(img_path,des_dir )
            # except:
            #     pass
            
            # try:
            #     if torch.any(results[0].boxes.cls==4.) :  #  判断一个数是否在PyTorch张量中,torch.any(tensor == number)
            #         des_dir='/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230622/20230622_SLA'
            #         mkdir(des_dir)
            #         shutil.move(img_path,des_dir ) 
            # except:
            #     pass
            
            # try:
            #     if torch.any(results[0].boxes.cls==5.) :  #  判断一个数是否在PyTorch张量中,torch.any(tensor == number)
            #         des_dir='/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230622/20230622_SRA'       
            #         mkdir(des_dir)
            #         shutil.move(img_path,des_dir )           
            # except:
            #     pass

            # try:
            #     if len(results[0].boxes.cls)>0 and (results[0].boxes.cls == 1.).all().item():  #  使用(tensor == value).all()来判断张量中的所有元素是否都等于给定的值
            #         des_dir='/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230622/20230622_onlyPC'   
            #         mkdir(des_dir)
            #         shutil.move(img_path,des_dir )
            # except:
            #     pass

            # try:
            #     if ( len(results[0].boxes.cls)>0 and (results[0].boxes.cls == 2.).all().item() ) or  \
            #                 ( (torch.any(results[0].boxes.cls==1.) ) and (torch.any(results[0].boxes.cls==2.))and (results[0].boxes.shape[0] ==2) ):  #判断所有的元素都为2,或者只有1和2,两者混合存在
            #         des_dir='/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230622/20230622__onlySA' 
            #         mkdir(des_dir)
            #         shutil.move(img_path,des_dir )        
            # except:
            #     pass
            
            # try:
            #     des_dir='/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230622/20230622_other' 
            #     mkdir(des_dir)
            #     shutil.move(img_path,des_dir )
            # except:
            #     pass
            
        return results
