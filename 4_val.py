# from ultralytics import YOLO

# def predict():
#     # Load a model
#     # model = YOLO('yolov8n.pt')  # 加载官方的模型权重作评估
#     model = YOLO('/home/xnwu/wangyong/Code/Yolov8/runs/detect/yolov8s_train_on_dataset8_crop_side/weights/best.pt')  # 加载自定义的模型权重作评估

#    	# 评估
#     imgsz=320
#     metrics = model.val(imgsz=320)  # 不需要传参，这里定义的模型会自动在训练的数据集上作评估
#     print("======>imsize:",imgsz)
#     print(metrics.box.map)  # map50-95
#     print(metrics.box.map50)  # map50
#     print(metrics.box.map75)  # map75
#     print(metrics.box.maps)  # 包含每个类别的map50-95列表

# predict()
import torch ,thop
from copy import deepcopy
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, __version__
from pathlib import Path
from ultralytics import YOLO
import torch.nn as nn


def is_parallel(model):
    """Returns True if model is of type DP or DDP."""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))
def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model

def get_flops(model, imgsz=640):
    """Return a YOLO model's FLOPs."""
    try:
        model = de_parallel(model)
        p = next(model.parameters())
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32  # max stride
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
        flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1E9 * 2 if thop else 0  # stride GFLOPs
        imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]  # expand if int/float
        return flops * imgsz[0] / stride * imgsz[1] / stride  # 640x640 GFLOPs
    except Exception:
        return 0
def get_num_params(model):
    """Return the total number of parameters in a YOLO model."""
    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model):
    """Return the total number of parameters with gradients in a YOLO model."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)

def model_info(model, detailed=False, verbose=True, imgsz=640):
    """
    Model information.

    imgsz may be int or list, i.e. imgsz=640 or imgsz=[640, 320].
    """
    if not verbose:
        return
    n_p = get_num_params(model)  # number of parameters
    n_g = get_num_gradients(model)  # number of gradients
    n_l = len(list(model.modules()))  # number of layers
    if detailed:
        LOGGER.info(
            f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            LOGGER.info('%5g %40s %9s %12g %20s %10.3g %10.3g %10s' %
                        (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std(), p.dtype))

    # imgsz=[320,640]    #tag：修改输入尺寸为指定大小，模型计算量的计算需要输入尺寸
    flops = get_flops(model, imgsz)
    # flops = thop.profile(model, inputs=320, verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
    print('================>',flops)
    fused = ' (fused)' if getattr(model, 'is_fused', lambda: False)() else ''
    fs = f', {flops:.1f} GFLOPs' if flops else ''
    yaml_file = getattr(model, 'yaml_file', '') or getattr(model, 'yaml', {}).get('yaml_file', '')
    model_name = Path(yaml_file).stem.replace('yolo', 'YOLO') or 'Model'
    LOGGER.info(f'{model_name} summary{fused}: {n_l} layers, {n_p} parameters, {n_g} gradients{fs}')
    return n_l, n_p, n_g, flops

model = YOLO('/home/xnwu/wangyong/Code/Yolov8/runs/detect/yolov8n_train_dataset8_new_crop_side/weights/best.pt')  # 加载官方的模型权重作评估
model_info(model, imgsz=640)
