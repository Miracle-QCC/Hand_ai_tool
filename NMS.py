import torch
import numpy as np

def iou(box1,box2):

    xmin = np.maximum(box1[0],box2[:,0])
    ymin = np.maximum(box1[1],box2[:,1])

    xmax = np.minimum(box1[2],box2[:,2])
    ymax = np.maximum(box1[3],box2[:,3])

    inter = torch.clamp(xmax - xmin + 1,min=0) * torch.clamp(ymax - ymin + 1,min=0)
    union = (box1[2] - box1[0] + 1) * (box1[3] - box1[:1] + 1) + (box2[:,2] - box2[:,0] + 1) * (box2[:,3] - box2[:,1] + 1) - inter

    iou = inter / (union + 1e-5)
    return iou

# boxes(n,nums,4+1+classnums)
def NMS(boxes,conf_thres = 0.5,nms_thres=0.4):
        bs = boxes.shape[0]
        output_boxes = []
        for i in range(bs):
            pred = boxes[i] # nums,4+1+class_num
            score = pred[:,4]
            mask = score > conf_thres # 存在物体

            true_dets = pred[mask]


            # 获取每个框的类别预测信息
            class_scores = true_dets[:,5:]

            class_conf,class_pred = torch.max(class_scores,dim=-1,keepdim=True)

            detections = torch.cat([true_dets[:,:4],class_conf,class_pred],dim=-1)
            unique_class = torch.unique(detections[:,-1])
            if len(unique_class) == 0:
                continue

            best_box = []
            for c in unique_class:
                cls_mask = detections[:,-1] == c

                detection = detections[cls_mask]
                scores = detection[:,4]

                # 逆序
                arg_sort = torch.argsort(-scores)
                detection = detection[arg_sort]

                while len(detection) != 0:
                    best_box.append(detection[0])
                    ious = iou(best_box[-1],detection)
                    iou_mask = ious < nms_thres
                    detection = detection[iou_mask]
            output_boxes.append(best_box)
        print(output_boxes)
if __name__ == '__main__':
    boxes = torch.FloatTensor([[0,0,1,1,0.7,0.8,0.2],
                               [0.5,0,1.5,1,0.7,0.6,0.7]]).reshape(1,2,-1)

    NMS(boxes)

    """
    ### 总结一下，NMS需要首先实现一个IOU计算函数
    
    ### 然后对n,nums,4+1+class_num的tensor进行处理
    
    ### 按照batch_size维度进行处理
    1.detections = boxes[bs]
    2.根据置信度mask掉背景
        cls_mask = detections[:,4] < conf_thres
        detections = detections[cls_mask]
    
    3.对剩下的框的类别预测进行排序，利用torch.max函数，并且保持维度
        class_conf,class_pred = tor.max(detections[:,5:],dim=-1,keep_dim=True)
        
    4.拼接成新的boxes
        detections = torch.cat([detections[:,:4],class_conf,class_pred],dim=-1)
        
    4.统计类别总数
        unique_classes = torch.unique(class_pred)
        if len(unique_classes) == 0:
            continue
    
    5.针对每个类进行NMS
    best_box = []
    for c in unique_classes:
        class_mask = detections[:,-1] == c
        detection = detections[class_mask]
        
        scores = detection[:,-2]
        scores_idx = torch.argsort(-scores)
        
        detection = detection[scores_idx]
        
        while len(detection) != 0:
            best_box.append(detection[0])
            ious = iou(best_box[-1],detection)
            iou_mask = ious < nms_thres
            detection = detection[iou_mask]
        output.append(best_box)
    
    
    """




