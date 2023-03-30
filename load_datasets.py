import torch
import torchvision
from torchvision import datasets
from torchvision import transforms

from pathlib import Path

base_dir = Path(Path.cwd())
save_dir = base_dir / 'datasets'

train_transform = transforms.Compose([transforms.RandomCrop((224,224)),
                                      transforms.RandomVerticalFlip(),
                                      transforms.RandomRotation([0, 270]),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])
test_transform = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor()])

train_set = datasets.ImageNet(root = save_dir,
                              train=True,
                              download=True,
                              transform=train_transform)

test_set = datasets.ImageNet(root = save_dir,
                             train=True,
                             download=True,
                             transform=test_transform)


