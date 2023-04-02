import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

class VOCDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        
        self.boxes = []
        self.images = []
        for image, text in tqdm(zip(self.annotations['image'], self.annotations['text'])):
            label_path = os.path.join(self.label_dir, text)
            img_path = os.path.join(self.img_dir, image)
            box = []
            with open(label_path) as f:
                for line in f.readlines():
                    label, x, y, w, h = [float(x) if float(x) != int(float(x)) else int(x)
                                        for x in line.replace('\n', '').split()]
                    box.append([label, x, y, w, h])
                self.boxes.append(box)
            self.images.append(Image.open(img_path))

        

    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index):
        # label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        # boxes = []
        # with open(label_path) as f:
        #     for line in f.readlines():
        #         label, x, y, w, h = [float(x) if float(x) != int(float(x)) else int(x)
        #                              for x in line.replace('\n', '').split()]
        #         boxes.append([label, x, y, w, h])

        # img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = self.images[index]
        boxes = torch.tensor(self.boxes[index])

        if self.transform:
            image = self.transform(image)
        
        target = torch.zeros((self.S, self.S, self.C + 5*self.B))
        for box in boxes:
            # 이미지 전체 크기 S를 기준 x, y, w, h
            # 0 <= x, y, w, h <= 1
            label, x, y, w, h = box.tolist()
            label = int(label)

            # Grid Cell의 행(i)과 열(j)
            i, j = int(self.S*y), int(self.S*x)
            # Cell 안에서의 x, y 위치(0 <=)
            x, y = self.S*x - j, self.S*y-i
            # Cell의 크기 S를 기준으로 width(w)와 height(h)의 길이
            w, h = self.S*w, self.S*h

            if target[i, j, 20] == 0:
                target[i, j, 20] = 1
                
                coordinate = torch.tensor([x, y, w, h])
                target[i, j, 21:25] = coordinate

                target[i, j, label] = 1
                
        return image, target


if __name__ == '__main__':
    # a = torch.randn(1,5)

    # b, c, d, e, f = a[0].tolist()

    # print(b, c, d, e, f)

    base_dir = Path.cwd()
    data_dir = base_dir/'datasets'
    csv_file = data_dir / '100examples.csv'
    img_dir = data_dir / 'images'
    label_dir = data_dir / 'labels'
    log_dir = base_dir / 'checkpoints'

    a = VOCDataset(csv_file, img_dir, label_dir, S=7, B=1, C=20, transform=None)
    print(next(iter(a)))