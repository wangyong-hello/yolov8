import argparse

import cv2
import numpy as np
import onnxruntime as ort
import torch
import time
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml

def xywh2xyxy(x, y, w, h):
    #  dw = x[..., 2] / 2  # half-width
    # dh = x[..., 3] / 2  # half-height
    # y[..., 0] = x[..., 0] - dw  # top left x
    # y[..., 1] = x[..., 1] - dh  # top left y
    # y[..., 2] = x[..., 0] + dw  # bottom right x
    # y[..., 3] = x[..., 1] + dh  # bottom right y
    dw=w/2
    dh=h/2
    xmin=x-dw
    ymin=y-dh
    xmax=x+dw
    ymax=y+dh
    return [xmin,ymin,xmax,ymax]
class YOLOv8:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model, confidence_thres, iou_thres):
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.input_img=None

        # Load the class names from the COCO dataset
        self.classes = yaml_load(check_yaml('/home/xnwu/wangyong/Code/Yolov8/ultralytics/cfg/score_data.yaml'))['names']

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, score, class_id):
       
        # Extract the coordinates of the bounding box
        x1, y1, x2, y2 = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]
        
        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int( x2), int(  y2)), color, 2)

        # Create the label text with class name and score
        label = f'{self.classes[class_id]}: {score:.2f}'
        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)
        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        print(label)

    def letterBox(self,input_img,new_shape,new_height,new_width):
        r = min(new_shape[0] / new_height, new_shape[1] / new_width)
        # ratio = r, r
        new_unpad = int(round(new_width * r)), int(round(new_height * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        # dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding，矩形推理
        dw /= 2  
        dh /= 2
        if (new_height,new_width) != new_unpad:  # resize
            input_img = cv2.resize(input_img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) , int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) , int(round(dw + 0.1))
        img = cv2.copyMakeBorder(input_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border
        # cv2.imshow('image', img)
        # if cv2.waitKey(0) & 0xFF==ord('q'):
        #     pass
        return img
    
    def scale_box(self,xyxy,crop_img,new_shape):
        gain = min( new_shape[0] /crop_img.shape[0],  new_shape[1]/ crop_img.shape[1])
        pad = round((new_shape[1] - crop_img.shape[1] * gain) / 2 - 0.1), round(
            (new_shape[0] - crop_img.shape[0] * gain) / 2 - 0.1)  # wh padding
        # print('fgds')
        box=xyxy
        box[0] -= pad[0]  # x padding
        box[2] -= pad[0]  # x padding
        box[1] -= pad[1]  # y padding
        box[3] -= pad[1]  # y padding
        box[0] /= gain
        box[1] /= gain
        box[2] /= gain
        box[3] /= gain
        
        return box
    def preprocess(self,input_img):
        self.input_img = input_img
        # Get the height and width of the input image
        self.img_height, self.img_width = self.input_img.shape[:2]
        wid_l_ratio = 0.25
        wid_r_ratio = 0.25
        h_t_ratio = 0.4
        h_bot_ratio = 0.15
        x_min = int( wid_l_ratio * self.img_width )
        y_min = int(h_t_ratio * self.img_height )
        x_max = int(self.img_width -  wid_r_ratio * self.img_width )
        y_max = int(self.img_height - h_bot_ratio * self.img_height )
        crop_img=self.input_img[y_min:y_max, x_min:x_max]
        # cv2.imshow('precrocess img', crop_img)  
        # k = cv2.waitKey(100) & 0xFF
        # if k == 27: # wait for ESC key to exit   #按esc退出，下一张
        #     cv2.destroyAllWindows()
        self.input_img=crop_img
        new_height=int(y_max-y_min)
        new_width=int(x_max-x_min)
        self.img_width=new_width
        self.img_height=new_height
        
        # image = np.zeros((length, length, 3), np.uint8)
        # image[0:new_height, 0:new_width] = self.input_img

        new_shape=[192,320]
        image=self.letterBox(crop_img,new_shape,new_height,new_width)
        # cv2.imshow('precrocess img', image)  
        # k = cv2.waitKey(100) & 0xFF
        # if k == 27: # wait for ESC key to exit   #按esc退出，下一张
        #     cv2.destroyAllWindows()

        image_data = np.array(image).astype(np.float32) / 255.0
        image_data=image_data[..., ::-1]     #    BGR2RGB

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0)

        # Return the preprocessed image data
        return image_data,crop_img

    def postprocess(self, crop_img, output):
        
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []
     
        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                new_shape=[192,320]
                
                xyxy=xywh2xyxy( x, y, w, h)
                
                box=self.scale_box(xyxy,crop_img,new_shape)
                #xmin,ymin,xmax,ymax
                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                # boxes.append([left, top, width, height])
                boxes.append([int(box[0]),int(box[1]),int(box[2]), int(box[3])])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)  
        # torchvision.ops.nms(boxes, scores, self.iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            self.draw_detections(crop_img, box, score, class_id)

        # Return the modified input image
        return crop_img

    def main(self,input_img):
      
        # Create an inference session using the ONNX model and specify execution providers
        session = ort.InferenceSession(self.onnx_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        # Get the model inputs
        model_inputs = session.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # Preprocess the image data
        img_data,crop_img = self.preprocess(input_img)

        # Run inference using the preprocessed image data
        outputs = session.run(None, {model_inputs[0].name: img_data})

        # Perform post-processing on the outputs to obtain output image.
        return self.postprocess(crop_img, outputs)  # output image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/home/xnwu/wangyong/Code/Yolov8/runs/detect/yolov8n_train_dataset8_new_norect_no_fliplr_no_scale/weights/best.onnx', help='Input your ONNX model.')
    parser.add_argument('--source', type=str, default="/media/xnwu/2AC0DAF3C0DAC3EB/Datasets/DVR/data/20231209_1211/2023-12-11-08-12-50.mp4", help='Path to input image.')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='NMS IoU threshold')
    args = parser.parse_args()
    #'/home/xnwu/wangyong/Code/Yolov8/official_weights/20230623100932559_LGWEF6A75MH250240_0_0_0_24.jpg

    # Check the requirements and select the appropriate backend (CPU or GPU)
    check_requirements('onnxruntime-gpu')
    # check_requirements('onnxruntime-gpu' if torch.cuda.is_available() else 'onnxruntime')
    # check_requirements

    # Create an instance of the YOLOv8 class with the specified arguments
    detection = YOLOv8(args.model, args.conf_thres, args.iou_thres)

    # Perform object detection and obtain the output image
    suffixStr = args.source.split(".")[-1]
    if(suffixStr == "jpg" or suffixStr == "png" or suffixStr == "JPG" or suffixStr == "PMG" or suffixStr == "bmp" or suffixStr == "jpeg"):
        original_image: np.ndarray = cv2.imread(args.source)
        output_image = detection.main(original_image)
        cv2.imshow('image', output_image)
        if cv2.waitKey(0) & 0xFF==ord('q'):
            cv2.destroyAllWindows()
            
    elif (suffixStr == "mp4" or suffixStr == "mp4" or suffixStr == "avi" or suffixStr == "H264" or suffixStr == "h264"):
        cap = cv2.VideoCapture(args.source)
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if (ret==True ):
                if (frame_id >= 0 and frame_id%1==0):
                    time1=time.time()
                    output_image = detection.main(frame)
                    cv2.imshow('image', output_image)
                    if cv2.waitKey(0) & 0xFF==ord('q'):
                        pass
                    time2=time.time()
                    print(f'frame ID{frame_id}，推理时间为:{time2}')
                    
                else:
                    # print('frame ID',frame_id)
                    pass
                frame_id += 1
            else:
                break
        cap.release()
