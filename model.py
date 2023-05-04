import torch
import torch.nn as nn

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'netB': [32, 32, 32, 32, 'M', 64, 64, 64, 64, 'M']
}


def make_features(cfg: list):
    layer = []
    in_chanels = 36
    for v in cfg:
        if v == 'M':
            layer += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layer += [nn.Conv2d(in_chanels, v, 3, 1, 1), nn.BatchNorm2d(v), nn.ReLU(True)]
            in_chanels = v
    return nn.Sequential(*layer)  # 列表、元组前面加星号作用是将列表解开成独立的参数，传入函数


class CNN(nn.Module):
    def __init__(self, feature, num_classes=1000):
        super(CNN, self).__init__()
        self.feature = feature
        self.classifier = nn.Sequential(
            nn.Linear(64*2*2, 250),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(250, num_classes),
            # nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


def cnn(model_name='vgg16', **kwargs):
    cfg = []
    try:
        cfg = cfgs[model_name]
    except ImportError:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = CNN(make_features(cfg), num_classes=2)
    return model


net = cnn('netB')
# net.classifier[0].weight.data.normal_(0, 0.01)
# net.classifier[3].weight.data.normal_(0, 0.01)
# net = cnn('vgg16')
