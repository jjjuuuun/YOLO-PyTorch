import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

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

def mAP(pred_boxes, true_boxes, C, iou_threshold=0.5):
    ap = [] # Average Precision
    epsilon = 1e-6

    for c in range(C):
        detections = [detection for detection in pred_boxes if detection[1] == c]
        ground_truths = [true_boxe for true_boxe in true_boxes if true_boxe[1] == c]

        # 각 image에서 class가 c인 ground truth box 수
        # train_idx = 0이고 train_idx=0인 bbox가 2개라면 n_box[0] = 2가 된다.
        # n_bbox = {0:2, 1:5}
        n_bbox = Counter([gt[0] for gt in ground_truths])

        # n_bbox = {0:torch.tensor[0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in n_bbox.items():
            n_bbox[key] = torch.zeros(val)

        # Confidence score에 따라 내림차순 정렬
        detections = sorted(detections, key=lambda conf : conf[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bbox = len(ground_truths) # batch_size * S * S에서 class가 c인 bbox 개수

        if total_true_bbox == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # 현재 박스(detection)와 같은 이미지에 있는 ground truths bbox들
            ground_truths_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]
            
            n_gts = len(ground_truths_img)

            # 현재 박스(detection)와 같은 이미지에 있는 ground truths bbox들 중 가장 큰 iou를 갖는 bbox
            best_iou = 0
            for idx, gt in enumerate(ground_truths_img):
                iou = IoU(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:])
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            # IoU가 iou_threshold보다 큰 경우
            if best_iou > iou_threshold:
                # 내림차순 정렬 했으므로 처음 나온 것은 TP
                if n_bbox[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    n_bbox[detection[0]][best_gt_idx] = 1
                # 같은 것을 한 번 더 예측한 것은 잘 못 예측한 것 (예측은 한 번 만 해야하므로)
                else:
                    FP[detection_idx] = 1
            # IoU가 iou_threshold보다 작은 경우
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim = 0)
        FP_cumsum = torch.cumsum(FP, dim = 0)
        recall = TP_cumsum / (total_true_bbox + epsilon)
        recall = torch.cat((torch.tensor([0]), recall))
        precision = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precision = torch.cat((torch.tensor([1]), precision))

        # torch.trapz for numerical integration
        # 넓이 구하기
        ap.append(torch.trapz(precision, recall))

    return sum(ap) / len(ap)


def cell_to_boxes(output, S):
    convert_output = output.reshape(output.shape[0], S*S, -1)
    convert_output[..., 0] = convert_output[..., 0].long()
    # all_bboxes = []

    bboxes = []
    img_idx = 0
    for batch in range(output.shape[0]):
        for idx in range(S*S):
            bboxes.append([img_idx]+[x.item() for x in convert_output[batch, idx, :]])
        img_idx += 1
        # all_bboxes.append(bboxes)
    # return all_bboxes
    return bboxes

def choice_boxes(img, output, S, C, B, count='1ofB', scale=False):
    batch_size = output.shape[0]
    output = output.reshape(batch_size, S, S, C+5*B)
    label = output[..., :20].argmax(-1).unsqueeze(-1)
    if count == 'BofB':
        if scale:
            output[...,21:25] = convert_scale(output[...,21:25], img)
            output[...,26:30] = convert_scale(output[...,26:30], img)
        return torch.cat((label, output[...,20:]), dim=-1)
    else:
        if count == '1ofB':
            scores = torch.cat(
                (output[..., 20].unsqueeze(0), output[..., 25].unsqueeze(0)), dim=0
            )
            bbox1 = output[..., 21:25]
            bbox2 = output[..., 26:30]
            score, idx = torch.max(scores, dim=0)
            score, idx = score.unsqueeze(-1), idx.unsqueeze(-1)
            bbox = (1 - idx) * bbox1 + idx * bbox2
        elif count == '1of1':
            score = output[..., 20:21]
            bbox = output[..., 21:25]

        if scale:
            bbox = convert_scale(bbox, img)

        return torch.cat((label, score, bbox), dim = -1)


def convert_scale(bbox, img):
    """bounding box의 scale을 이미지에 맞게 조정

    Args:
        output (_type_): bounding box : [batch, S, S, 4]
        img (_type_): [batch, 3, 448, 448]
        S (_type_): 7
    """
    batch_size, S, _, vec_length = bbox.shape
    _, _, width, height = img.shape

    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)
    x = 1 / S * (bbox[..., :1] + cell_indices)
    y = 1 / S * (bbox[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w = 1 / S * bbox[..., 2:3]
    h = 1 / S * bbox[..., 3:4]

    bboxes = torch.cat((x,y,w,h), dim = -1)

    bboxes[...,0] = (bboxes[...,0] - bboxes[...,2] / 2) * width
    bboxes[...,1] = (bboxes[...,1] - bboxes[...,3] / 2) * height
    bboxes[...,2] = bboxes[...,2] * width
    bboxes[...,3] = bboxes[...,3] * height

    return bboxes

def plot_images(images, targets, S, B, count):
    for image, target in zip(images, targets):
        img = draw_image(image, target, S, B, count)
        plt.imshow(np.array(img))
        plt.show()

def draw_image(img, target, S, B, count):
    img = to_pil_image(img)
    draw = ImageDraw.Draw(img)
    colors = np.random.randint(0, 255, size=(80,3), dtype=np.uint8)
    font = ImageFont.truetype("C:\Windows\Fonts\Arial.ttf")

    if count == 'BofB':
        for i in range(S):
            for j in range(S):
                label, *bbox = target[i][j]
                for m in range(B):
                    score, x, y, w, h = bbox[m*5:(m+1)*5]
                    draw.rectangle(((x, y), 
                                    (x+w, y+h)), 
                                    outline=(0,0,0), 
                                    width = 3)
    else:
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

    return img    

if __name__ == '__main__':
    a = torch.randn(3, 7, 7, 6)
    b = torch.randn(3, 7, 7, 6)
    a = cell_to_boxes(a, 7)
    b = cell_to_boxes(b, 7)
    print(len(b)) # 49 x 3 = 147
    print(len(b[0])) # 7 >> [image idx, class, score, x, y, w, h]
    print(mAP(a, b, 20, iou_threshold=0.5))