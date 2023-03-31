import torch
import torch.nn as nn
import torchvision
from utils import *

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
        bbox_target = target[..., 21:25]

        iou1 = IoU(bbox1, bbox_target) # [batch, S, S, 1]
        iou2 = IoU(bbox2, bbox_target) # [batch, S, S, 1]
        ious = torch.cat([iou1.unsqueeze(0), iou2.unsqueeze(0)], dim=0) # [2, batch, S, S, 1]
        
        # 하나의 Cell에서 IoU가 가장 큰 Bounding Box 선택 (학습시 하나의 bounding box만 선택)
        iou_max, iou_idx = torch.max(ious, dim=0) # [batch, S, S, 1]
        
        # 1^{obj}_{i} : i번째 Cell 안에 object가 있는지 : c1, c2
        # Target(Ground Truth)
        exist_obj = target[..., 20].unsqueeze(3)


        # ================== #
        # LOCALIZATIONI LOSS #
        # ================== #

        # 1^{obj}_{ij}
        # Object가 있는 i번째 Cell의 responsible bounding box의 좌표
        # [batch, S, S, 4]
        responsible_coordinate = exist_obj * (
            iou_idx*pred[..., 26:30] + (1-iou_idx)*pred[..., 21:25]
        )

        # 1^{obj}_{ij}
        # Object가 있는 i번째 Cell의 Ground Truth Bounding box의 좌표
        # [batch, S, S, 4]
        groundtruth_coordinate = exist_obj * target[..., 20:25]

        # Paper : 
        # To partially address this we predict 
        # the square root of the bounding box width and height instead of the width and height directly.
        responsible_coordinate = torch.sign(responsible_coordinate[..., 2:4]) * torch.sqrt(
            torch.abs(responsible_coordinate[...,2:4] + 1e-6)
        )
        groundtruth_coordinate = torch.sqrt(groundtruth_coordinate[...,2:4])

        # Bounding Box Regression
        bbox_loss = self.mse(
            torch.flatten(responsible_coordinate, end_dim=-2),
            torch.flatten(groundtruth_coordinate, end_dim=-2)
        )


        # ================ #
        # CONFINDENCE LOSS #
        # ================ #

        # 1^{obj}_{ij}
        # Object가 있는 i번째 Cell의 responsible bounding box의 confidence score(objectness)
        # [batch, S, S, 1]
        responsible_obj = exist_obj * (
            iou_idx*pred[..., 25:26] + (1-iou_idx)*pred[..., 20:21]
        )
        # 1^{noobj}_{ij}
        # Object가 없는 i번째 Cell의 responsible bounding box의 confidence score(objectness)
        # [batch, S, S, 1]
        responsible_noobj = (1 - exist_obj) * (
            iou_idx*pred[..., 25:26] + (1-iou_idx)*pred[..., 20:21]
        )

        # 1^{obj}_{ij}
        # Object가 있는 i번째 Cell의 Ground Truth Bounding box의 confidence score(objectness)
        # [batch, S, S, 1]
        groundtruth_obj = exist_obj * target[..., 20:21]
        # 1^{noobj}_{ij}
        # Object가 없는 i번째 Cell의 Ground Truth Bounding box의 confidence score(objectness)
        # [batch, S, S, 1]
        groundtruth_noobj = (1- exist_obj) * target[..., 20:21]

        # Object Loss
        object_loss = self.mse(
            torch.flatten(responsible_obj*iou_max),
            torch.flatten(groundtruth_obj)
        )
        # No Object Loss
        no_object_loss = self.mse(
            torch.flatten(responsible_noobj*iou_max),
            torch.flatten(groundtruth_noobj)
        )


        # ========== #
        # CLASS LOSS #
        # ========== #

        # 1^{obj}_{i}
        # Object가 있는 i번째 Responsible Cell의 Class Probability
        # [batch, S, S, 20]
        responsible_class = exist_obj * (
            iou_idx*pred[..., :20] + (1-iou_idx)*pred[..., :20]
        )

        # 1^{obj}_{i}
        # Object가 있는 i번째 Ground Truth Cell의 Class Probability
        # [batch, S, S, 20]
        groundtruth_class = exist_obj * target[..., :20]

        # Bounding Box Regression
        class_loss = self.mse(
            torch.flatten(responsible_class, end_dim=-2),
            torch.flatten(groundtruth_class, end_dim=-2)
        )

        loss = (self.lambda_coord * bbox_loss +
                object_loss +
                self.lambda_noobj * no_object_loss +
                class_loss)

        return loss
        

        

        





import numpy as np
# a = torch.tensor(np.random.randint(1, 10, (1, 7, 7, 30)))
# b = torch.tensor(np.random.randint(1, 10, (1, 7, 7, 30)))
# c = torch.tensor(np.random.randint(1, 10, (1, 7, 7, 30)))
# bbox1 = a[..., 21:25]
# bbox2 = b[..., 21:25]
# bbox3 = c[..., 21:25]

# iou1 = IoU(bbox1, bbox2)
# iou2 = IoU(bbox1, bbox3)

# ious = torch.cat([iou1.unsqueeze(0), iou2.unsqueeze(0)], dim=0)

# val, idx = torch.max(ious, dim=0)
# print(val.shape)
# print(idx.shape)
# print(bbox1[0])
# bbox1[...,2] = bbox1[...,0] + bbox1[...,2]
# print(bbox1[0])

# print(bbox1.shape, bbox2.shape)
# ops.box_iou

a = list(range(0, 10))
b = list(range(10, 20))

print(a[:2:10])
