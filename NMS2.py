import torch
import numpy as np

def iou(box1,box2):
    x_min = np.maximum(box1[0],box2[:,0])
    y_min = np.maximum(box1[1],box2[:,1])

    x_max = np.minimum(box1[2],box2[:,2])
    y_max = np.maximum(box1[3],box2[:,3])

    inter = torch.clamp(x_max - x_min + 1,min=0) * torch.clamp(y_max - y_min+1,min=0)
    union = (box1[2] - box1[0] + 1) * (box1[3] - box1[1]) + (box2[:,2] - box2[:,0] + 1) * (box2[:,3] - box2[:,1] + 1) - inter

    iou = inter / (union + 1e-6)
    return iou

def NMS(boxes,conf_thres=0.5,nms_thres=0.4):
    bs = boxes.shape[0]
    output_box = []
    for i in range(bs):
        detections = boxes[i]

        conf_mask = detections[:,4] > conf_thres
        detections = detections[conf_mask]

        class_conf,class_pred = torch.max(detections[:,5:],dim=-1,keepdim=True)

        unique_class = torch.unique(class_pred)
        if len(unique_class) == 0:
            continue
        detections = torch.cat([detections[:,:4],class_conf,class_pred],dim=-1)
        best_box = []
        for c in unique_class:
            class_mask = detections[:,-1] == c
            detection = detections[class_mask]

            scores = detection[:,-2]
            scores_idx = torch.argsort(-scores)

            detection = detection[scores_idx]
            while len(detection):
                best_box.append(detection[0])
                ious = iou(best_box[-1],detection)

                iou_mask = ious < nms_thres

                detection = detection[iou_mask]
        output_box.append(best_box)

    print(output_box)

if __name__ == '__main__':
    boxes = torch.FloatTensor([[0, 0, 1, 1, 0.7, 0.8, 0.2],
                               [0.5, 0, 1.5, 1, 0.7, 0.6, 0.7]]).reshape(1, 2, -1)

    NMS(boxes)
