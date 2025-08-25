# mAP50,precision,recall
import numpy as np
import torch


def get_TrueBox_and_predBoex(targets, predicts):
    """

    :param targets:
    :param predicts:
    :return:
    """
    true_boxes = []
    pred_boxes = []
    for i in range(len(predicts)):
        socres_tensor = predicts[i]["scores"]
        sorted_indices = torch.argsort(socres_tensor, descending=True)
        temp = []
        for idx in sorted_indices:
            temp.append(predicts[i]["boxes"][idx.item()].cpu())
        pred_boxes.append(temp)
        true_boxes.append(targets[i]["boxes"].cpu())
    return pred_boxes, true_boxes


class DetectionMetrics(object):
    def __init__(self):
        self.detected_boxes = []
        self.true_boxes = []

    def reset(self):
        self.detected_boxes = []
        self.true_boxes = []

    def calculate_iou(self, gt_box, pred_box):
        # Assuming boxes are in [x_min, y_min, x_max, y_max] format
        gx1, gy1, gx2, gy2 = gt_box
        px1, py1, px2, py2 = pred_box
        intersection_x = max(0, min(gx2, px2) - max(gx1, px1))
        intersection_y = max(0, min(gy2, py2) - max(gy1, py1))

        intersection_area = intersection_x * intersection_y
        gt_box_area = (gx2 - gx1) * (gy2 - gy1)
        pred_box_area = (px2 - px1) * (py2 - py1)

        iou = intersection_area / (gt_box_area + pred_box_area - intersection_area)
        return iou

    def addBatch(self, predicted_boxes, true_boxes):
        self.detected_boxes += predicted_boxes
        self.true_boxes += true_boxes

    def calculate_precision_recall(self, iou_threshold):
        # 计算precision和recall
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for detected, true in zip(self.detected_boxes, self.true_boxes):
            if isinstance(detected, list):
                detected = np.array([t.numpy() for t in detected])
            else:
                detected = detected.numpy() 
            if isinstance(true, list):
                true = np.array([t.numpy() for t in true])
            else:
                true = true.numpy()  
            used_box = []

            for pred_box in detected:
                match_found = False
                for true_box in true:
                    if self.calculate_iou(pred_box, true_box) >= iou_threshold:
                        match_found = True
                        used_box.append(true_box)
                        break
                if match_found:
                    true_positives += 1
                else:
                    false_positives += 1

            used_box = set(tuple(arr) for arr in used_box)
            false_negatives += (len(true) - len(used_box))
        print(true_positives, false_negatives,false_negatives)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        return precision, recall

    def calculate_map(self, iou_threshold):
        mAP, _ = self.calculate_precision_recall(iou_threshold=iou_threshold)
        return mAP
