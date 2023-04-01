import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont

CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

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

def convert_cell(pred, S, C, B):
    batch_size= pred.size(0)
    pred = pred.reshape(batch_size, S, S, C+5*B)
    return pred

def convert_scale(output, img, S):
    label = output[..., :20].argmax(-1).unsqueeze(-1)
    score = output[..., 20:21]
    bbox = output[..., 21:25]
    batch_size, _, width, height = img.shape

    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (bbox[..., :1] + cell_indices)
    y = 1 / S * (bbox[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w = 1 / S * bbox[..., 2:3]
    h = 1 / S * bbox[..., 3:4]
    print(score.shape, x.shape, w.shape)
    convert_bbox = torch.cat((label, score, x,y,w,h), dim = -1)

    convert_bbox[...,2] = (convert_bbox[...,2] - convert_bbox[...,4] / 2) * width
    convert_bbox[...,3] = (convert_bbox[...,3] - convert_bbox[...,5] / 2) * height
    convert_bbox[...,4] = convert_bbox[...,4] * width
    convert_bbox[...,5] = convert_bbox[...,5] * height

    return convert_bbox


def plot_image(image, target, S):
    """Plots predicted bounding boxes on the image"""
    img = to_pil_image(image)
    draw = ImageDraw.Draw(img)
    colors = np.random.randint(0, 255, size=(80,3), dtype=np.uint8)
    font = ImageFont.truetype("C:\Windows\Fonts\Arial.ttf")
    
    # Create a Rectangle potch
    for i in range(S):
        for j in range(S):
            label, score, x,y,w,h = target[i][j]
            if score != 0:
                color = colors[int(label)]
                name = CLASSES[int(label)]
                draw.rectangle(((x, y), 
                                (x+w, y+h)), 
                                outline=tuple(color), 
                                width = 3)
                

                tw, th = font.getsize(name)
                draw.rectangle(((x, y), 
                                (x+tw, y+th)),
                                fill = tuple(color + [0]))
                draw.text((x, y), name, fill=(255,255,255,0), font=font, align='center')

    # plt.imshow(np.array(img))
    # plt.show()
    return img    