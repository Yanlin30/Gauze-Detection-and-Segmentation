
import torch.nn as nn
import torchvision.models as models
import torch

class ssd(nn.Module):
    def __init__(self):
        super(ssd, self).__init__()
        self.model = models.detection.ssd300_vgg16(pretrianted=True)
        self.model.head.classification_head.num_classes = 2

    def forward(self, x, targets=None):
        if self.training and targets is not None:
            loss_dict = self.model(x, targets)
            return loss_dict
        else:
            return self.model(x)
