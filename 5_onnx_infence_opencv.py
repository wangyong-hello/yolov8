import argparse

import cv2.dnn
import numpy as np
from ultralytics.utils import  yaml_load
from ultralytics.utils.checks import check_yaml
import time

CLASSES = yaml_load(check_yaml('/home/xnwu/wangyong/Code/Yolov8/ultralytics/cfg/score_data.yaml'))['names']
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(onnx_model, original_image):
    
    # Load the ONNX model
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)

    # Read the input image
   
    [height, width, _] = original_image.shape
    
    wid_l_ratio = 0.25
    wid_r_ratio = 0.25
    h_t_ratio = 0.4
    h_bot_ratio = 0.15
    x_min = int( wid_l_ratio * width )
    y_min = int(h_t_ratio * height )
    x_max = int(width -  wid_r_ratio * width )
    y_max = int(height - h_bot_ratio * height )
    original_image=original_image[y_min:y_max, x_min:x_max]
    new_height=int(y_max-y_min)
    new_width=int(x_max-x_min)
    
    # Prepare a square image for inference
    length = max((new_height, new_width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:new_height, 0:new_width] = original_image

    scale = length / 320

    # Preprocess the image and prepare blob for model
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(320, 320), swapRB=True)
    model.setInput(blob)

    # Perform inference
    outputs = model.forward()

    # Prepare output array
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    # Apply NMS (Non-maximum suppression)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []

    # Iterate through NMS results to draw bounding boxes and labels
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            'class_id': class_ids[index],
            'class_name': CLASSES[class_ids[index]],
            'confidence': scores[index],
            'box': box,
            'scale': scale}
        detections.append(detection)
        draw_bounding_box(original_image, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                          round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))

    # Display the image with bounding boxes
    cv2.imshow('image', original_image)
    if cv2.waitKey(0) & 0xFF==ord('q'):
        pass
        # cv2.destroyAllWindows()
    return detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/home/xnwu/wangyong/Code/Yolov8/runs/detect/yolov8n_train_dataset8_no_fliplr_no_scale_no_rect_no_translate/weights/best.onnx', help='Input your ONNX model.')
    parser.add_argument('--source', default='/home/xnwu/wangyong/Dataset/test/20230718153007805_LGWEF6A75MH250240_0_0_0.mp4', help='Path to input image.')
    args = parser.parse_args()
    suffixStr = args.source.split(".")[-1]
    if(suffixStr == "jpg" or suffixStr == "png" or suffixStr == "JPG" or suffixStr == "PMG" or suffixStr == "bmp" or suffixStr == "jpeg"):
        original_image: np.ndarray = cv2.imread(args.source)
        main(args.model,original_image )
    
    elif (suffixStr == "mp4" or suffixStr == "mp4" or suffixStr == "avi" or suffixStr == "H264" or suffixStr == "h264"):
        cap = cv2.VideoCapture(args.source)
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if (ret==True ):
                if (frame_id >= 0 and frame_id%10==0):
                    # cv2.imread(frame)
                    # frame = cv2.resize(frame,(1280,720))
                    time1=time.time()
                    main(args.model,frame)
                    time2=time.time()
                    print(f'frame ID{frame_id}，推理时间为:{time2}')
                    
                else:
                    # print('frame ID',frame_id)
                    pass
                frame_id += 1
            else:
                break
        cap.release()
