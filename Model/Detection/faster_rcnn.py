
import torch.nn as nn
import torchvision.models as models


class Faster_RCNN(nn.Module):
    def __init__(self, model_type,num_classes=1):
        super(Faster_RCNN, self).__init__()
        if model_type == 50:
            self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_class=num_classes)
        else:
            self.model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True, num_class=num_classes)


    def forward(self, x, targets=None):
        if self.training and targets is not None:
            loss_dict = self.model(x, targets)
            return loss_dict
        else:
            return self.model(x)

