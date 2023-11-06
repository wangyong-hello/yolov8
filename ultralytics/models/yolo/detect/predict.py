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
            #     shutil.move(img_path,dst='/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/LOOP/1LOOP_for_objdet_noObj' )
            # if torch.any(results[0].boxes.cls==1.) and  results[0].boxes.shape[0]==1:  #  判断一个数是否在PyTorch张量中,torch.any(tensor == number)
            # #     shutil.move(img_path,dst= )    
            #     shutil.move(img_path,dst='/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/LOOP/1LOOP_for_objdet_onlyPC' )
            # if torch.any(results[0].boxes.cls==2.) and  results[0].boxes.shape[0]==1:  #  判断一个数是否在PyTorch张量中,torch.any(tensor == number)
            # #     shutil.move(img_path,dst= )    
            #     shutil.move(img_path,dst='/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/LOOP/1LOOP_for_objdet_onlySA' )
            # #ta
        return results
