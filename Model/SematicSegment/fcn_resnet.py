import torch.nn as nn
import torchvision.models as models


class fcn(nn.Module):
    def __init__(self, model_type=50, num_classes=2):
        super(fcn, self).__init__()
        if model_type == 50:
            self.model = models.segmentation.fcn_resnet50(pretrained=True)
        elif model_type == 101:
            self.model = models.segmentation.fcn_resnet101(pretrained=True)
        self.model.backbone.conv1 = nn.Conv2d(3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        # x = self.log_softmax(x['out'])  # 如果用NLLLoss需要加log
        x = self.softmax(x['out'])
        return x
