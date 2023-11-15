# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops

import shutil,torch

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
            #     shutil.move(img_path,dst='/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230614/20230614_for_obj_det_noObj' )
            if torch.any(results[0].boxes.cls==0.) :  #  判断一个数是否在PyTorch张量中,torch.any(tensor == number)
                    shutil.move(img_path,dst='/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230614/20230614_for_obj_det_LA' )
            try:
                if torch.any(results[0].boxes.cls==3.) :  #  判断一个数是否在PyTorch张量中,torch.any(tensor == number)
                    shutil.move(img_path,dst='/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230614/20230614_for_obj_det_RA' )
            except:
                pass
            
            try:
                if torch.any(results[0].boxes.cls==4.) :  #  判断一个数是否在PyTorch张量中,torch.any(tensor == number)
                    shutil.move(img_path,dst='/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230614/20230614_for_obj_det_SLA' )
            except:
                pass
            
            try:
                if torch.any(results[0].boxes.cls==5.) :  #  判断一个数是否在PyTorch张量中,torch.any(tensor == number)
                    shutil.move(img_path,dst='/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230614/20230614_for_obj_det_SRA' )
            except:
                pass
            # if len(results[0].boxes.cls)>0 and (results[0].boxes.cls == 1.).all().item():  #  使用(tensor == value).all()来判断张量中的所有元素是否都等于给定的值
            #     shutil.move(img_path,dst='/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230614/20230614_for_obj_det_onlyPC' )
           
            # if ( len(results[0].boxes.cls)>0 and (results[0].boxes.cls == 2.).all().item() ) or  \
            #             ( (torch.any(results[0].boxes.cls==1.) ) and (torch.any(results[0].boxes.cls==2.))and (results[0].boxes.shape[0] ==2) ):  #判断所有的元素都为2,或者只有1和2,两者混合存在
            #     shutil.move(img_path,dst='/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20230614/20230614_for_obj_det_onlySA' )
            
            #ta
        return results
