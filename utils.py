import torch

def IoU(bbox1, bbox2):
    """Intersection over Union

    Args:
        bbox1 (Tensor): Predicted Bounding Box -> (B X S X S X 4)
        bbox2 (Tensor): Target Bounding Box -> (B X S X S X 4)
    """
    bbox1_x1 = bbox1[...,0:1] - bbox1[...,2:3] / 2
    bbox1_y1 = bbox1[...,1:2] - bbox1[...,3:4] / 2
    bbox1_x2 = bbox1[...,2:3] + bbox1[...,2:3] / 2
    bbox1_y2 = bbox1[...,3:4] + bbox1[...,3:4] / 2
    bbox2_x1 = bbox2[...,0:1] - bbox2[...,2:3] / 2
    bbox2_y1 = bbox2[...,1:2] - bbox2[...,3:4] / 2
    bbox2_x2 = bbox2[...,2:3] + bbox2[...,2:3] / 2
    bbox2_y2 = bbox2[...,3:4] + bbox2[...,3:4] / 2

    x1 = torch.max(bbox1_x1, bbox2_x1)
    y1 = torch.max(bbox1_y1, bbox2_y1)
    x2 = torch.min(bbox1_x2, bbox2_x2)
    y2 = torch.min(bbox1_y2, bbox2_y2)

    # Intersection의 최솟값을 0으로 설정
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    bbox1_area = abs((bbox1_x2 - bbox1_x1) * (bbox1_y2 - bbox1_y1))
    bbox2_area = abs((bbox2_x2 - bbox2_x1) * (bbox2_y2 - bbox2_y1))

    return intersection / (bbox1_area + bbox2_area - intersection + 1e-6)
