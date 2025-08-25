
import torch.nn as nn
import torchvision.models as models


class r_fcn(nn.Module):
    def __init__(self,model_type, num_classes=1):
        super(r_fcn, self).__init__()
        if model_type == 50:
            self.model = models.segmentation.fcn_resnet101(pretrained=True)
        else:
            self.model = models.segmentation.fcn_resnet101(pretrained=True)
        self.model.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x['out'])
        return x

