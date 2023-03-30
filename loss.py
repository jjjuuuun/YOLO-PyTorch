import torch
import torch.nn as nn
import torchvision
from torchvision import ops

class YOLOLoss(nn.Module):

    def __init__(self, S=7,B=2,C=20):
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        self.mse = nn.MSELoss(reduction='sum')

        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, pred, target):
        pred = pred.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # pred : [range(0,20)][c1,x1,y1,w1,h1][c2,x2,y2,w2,h2]
        # target : [range(0,20)][c1,x1,y1,w1,h1]
        bbox1 = pred[..., 21:25] # Python Ellipsis 객체(한 번의 하나만 사용 가능)
        bbox2 = pred[..., 26:30]
        bbox_gt = target[..., 21:25]

        bbox1[...,2] = bbox1[...,0] + bbox1[...,2]
        bbox1[...,3] = bbox1[...,1] + bbox1[...,3]
        bbox2[...,2] = bbox2[...,0] + bbox2[...,2]
        bbox2[...,3] = bbox2[...,1] + bbox2[...,3]
        bbox_gt[...,2] = bbox_gt[...,0] + bbox_gt[...,2]
        bbox_gt[...,3] = bbox_gt[...,1] + bbox_gt[...,3]

        ops.box_iou(bbox1, )

import numpy as np
a = torch.tensor(np.random.randint(1, 10, (1, 7, 7, 30)))
b = torch.tensor(np.random.randint(1, 10, (1, 7, 7, 30)))
bbox1 = a[..., 21:25]
bbox2 = b[..., 21:25]

print(ops.box_iou(bbox1, bbox2))

# print(bbox1[0])
# bbox1[...,2] = bbox1[...,0] + bbox1[...,2]
# print(bbox1[0])

# print(bbox1.shape, bbox2.shape)
# ops.box_iou




