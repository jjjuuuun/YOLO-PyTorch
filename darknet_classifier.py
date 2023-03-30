import torch
import torch.nn as nn
from timm import list_models
from timm import create_model
from torchsummary.torchsummary import summary

# list_models('*', pretrained=True)
# net = create_model('darknet53', pretrained=True)
# summary(net, (3,448,448))
# print(net)

KERNEL_SIZE = {'init':7,
               'default':3,
               'down':1}

STRIDE = {'default':1,
          '2':2}

PADDING = {'default':0,
           '1':1,
           '3':3}

CONV_ARCHITECTURE = [
    [3,64,KERNEL_SIZE['init'],STRIDE['2'],PADDING['3'],'bnorm','max'],
    [64,192,KERNEL_SIZE['default'],STRIDE['default'],PADDING['1'],'bnorm','max'],
    [192,128,KERNEL_SIZE['down'],STRIDE['default'],PADDING['default'],'bnorm',None],
    [128,256,KERNEL_SIZE['default'],STRIDE['default'],PADDING['1'],'bnorm',None],
    [256,256,KERNEL_SIZE['down'],STRIDE['default'],PADDING['default'],'bnorm',None],
    [256,512,KERNEL_SIZE['default'],STRIDE['default'],PADDING['1'],'bnorm','max'],
    [512,256,KERNEL_SIZE['down'],STRIDE['default'],PADDING['default'],'bnorm',None],
    [256,512,KERNEL_SIZE['default'],STRIDE['default'],PADDING['1'],'bnorm',None],
    [512,256,KERNEL_SIZE['down'],STRIDE['default'],PADDING['default'],'bnorm',None],
    [256,512,KERNEL_SIZE['default'],STRIDE['default'],PADDING['1'],'bnorm',None],
    [512,256,KERNEL_SIZE['down'],STRIDE['default'],PADDING['default'],'bnorm',None],
    [256,512,KERNEL_SIZE['default'],STRIDE['default'],PADDING['1'],'bnorm',None],
    [512,256,KERNEL_SIZE['down'],STRIDE['default'],PADDING['default'],'bnorm',None],
    [256,512,KERNEL_SIZE['default'],STRIDE['default'],PADDING['1'],'bnorm',None],
    [512,512,KERNEL_SIZE['down'],STRIDE['default'],PADDING['default'],'bnorm',None],
    [512,1024,KERNEL_SIZE['default'],STRIDE['default'],PADDING['1'],'bnorm','max'],
    [1024,512,KERNEL_SIZE['down'],STRIDE['default'],PADDING['default'],'bnorm',None],
    [512,1024,KERNEL_SIZE['default'],STRIDE['default'],PADDING['1'],'bnorm',None],
    [1024,512,KERNEL_SIZE['down'],STRIDE['default'],PADDING['default'],'bnorm',None],
    [512,1024,KERNEL_SIZE['default'],STRIDE['default'],PADDING['1'],'bnorm',None],
    [1024,1024,KERNEL_SIZE['default'],STRIDE['default'],PADDING['1'],'bnorm',None],
    [1024,1024,KERNEL_SIZE['default'],STRIDE['2'],PADDING['1'],'bnorm',None],
    [1024,1024,KERNEL_SIZE['default'],STRIDE['default'],PADDING['1'],'bnorm',None],
    [1024,1024,KERNEL_SIZE['default'],STRIDE['default'],PADDING['1'],'bnorm',None]
    ]

class CNNBlock(nn.Module):
    def __init__(self, in_channles, out_channels, kernel_size, stride, padding, norm=None, pool=None):
        super(CNNBlock, self).__init__()
        layers = []
        layers += [nn.Conv2d(in_channles, out_channels, kernel_size, stride, padding)]
        if norm == 'bnorm':
            layers += [nn.BatchNorm2d(out_channels)]
        layers += [nn.LeakyReLU(0.1)]
        if pool == 'max':
            layers += [nn.MaxPool2d(2)]
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class BaseModel(nn.Module):
    """DarkNet Classifier

    Args:
        nn (_type_): _description_
    """
    def __init__(self, num_classes=0):
        super(BaseModel, self).__init__()
        # Classification Input size = (3, 224, 224)
        layers = []
        
        for order, conv_info in enumerate(CONV_ARCHITECTURE[:20]):
            layers += [CNNBlock(*conv_info)]

        self.pre_conv = nn.Sequential(*layers)
        self.Avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.Avg(x)
        x = torch.squeeze(x)
        x = self.fc1(x)

        return x
                                
class DarkNet(nn.Module):
    def __init__(self, split_size=7, num_boxes=2, num_classes=20, fc_dim=4096):
        super(DarkNet, self).__init__()
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        self.fc_dim = fc_dim

        layers = []
        for order, conv_info in enumerate(CONV_ARCHITECTURE):
            layers += [CNNBlock(*conv_info)]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv(x)

        channel_dim = x.size(1) # 1024

        x = nn.Flatten()(x)
        x = nn.Linear(self.S*self.S*channel_dim, self.fc_dim)(x)
        x = nn.Dropout(0.5)(x)
        x = nn.LeakyReLU(0.1)(x)
        x = nn.Linear(self.fc_dim, self.S*self.S*(self.C + self.B*5))(x)

        return x


# classifier = BaseModel(num_classes=1000)
detector = DarkNet(fc_dim=496)
# print(summary(classifier, (3,224,224)))
print(summary(detector, (3,448,448)))

