import torch
import numpy as np
import random
import os
from PIL import Image
from torch.utils.data  import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from loss import *
from utils import *
from datasets import *
from models import *
import matplotlib.pyplot as plt
import wandb
import torch.optim as optim
import time
from pathlib import Path

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

# CONSTANT VALUE
RANDOM_SEED = 223
BATCH_SIZE = 3
S = 7
C = 20
B = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0
EPOCHS = 1
TRAIN_VAL_SPLIT = 0.8
USE_WANDB = False


if USE_WANDB:
    print("Using wandb ...")
    # wandb.init(project=opt.wandb_project, entity=opt.wandb_entity)
    # wandb.config.update(opt)

print("Fixing Seed")
torch_seed(RANDOM_SEED)

base_dir = Path.cwd()
data_dir = base_dir/'datasets'
csv_file = data_dir / '100examples.csv'
img_dir = data_dir / 'images'
label_dir = data_dir / 'labels'
log_dir = base_dir / 'checkpoints'

if not log_dir.exists():
    print('Make log_dir')
    log_dir.mkdir()

transform = transforms.Compose([transforms.Resize((448,448)),
                                transforms.ToTensor()])
train_set, val_set = random_split(VOCDataset(csv_file, img_dir, label_dir, 
                                             S=7, B=1, C=20, transform=transform),
                                  [TRAIN_VAL_SPLIT, 1-TRAIN_VAL_SPLIT])
print(f'The number of training images = {len(train_set)}')
print(f'The number of validation images = {len(val_set)}')

train_iter = DataLoader(train_set,
                    batch_size=BATCH_SIZE,
                    shuffle = True)
val_iter = DataLoader(val_set,
                    batch_size=BATCH_SIZE,
                    shuffle = True)
print("Split training dataset and testing dataset")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'The device is ready\t>>\t{device}')

model = mdoel = DarkNet(S, C, B).to(device)
print('The model is ready ...')
print(summary(model, (3, 448, 448)))

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
# lr_scheduler
criterion = YOLOLoss(S, B, C)

print("Starting training ...")

for epoch in range(EPOCHS):
    start_time = time.time()
    train_epoch_loss = 0
    train_bboxes = []
    train_bbox = []

    model.train()
    for train_img, train_target in train_iter:
        train_img, train_target = train_img.to(device), train_target.to(device)

        optimizer.zero_grad()
        train_pred = model(train_img)

        train_iter_loss = criterion(train_pred, train_target)
        train_iter_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_iter_loss

        pred_bbox = choice_boxes(train_img, train_pred, S, C, B, count='1ofB', scale=False)
        target_bbox = choice_boxes(train_img, train_target, S, C, B=1, count='1of1', scale=False)
        
        pred_bbox = cell_to_boxes(pred_bbox, S)
        target_bbox = cell_to_boxes(target_bbox, S)
        train_mAP = mAP(pred_bbox, target_bbox, C, iou_threshold=0.5)
        print(f'{train_mAP:.4f}')

        break
    break
    #     # ======================================== #
    #     # Bounding Box가 어떻게 학습되고 있는지 확인 #
    #     pred_bbox = choice_boxes(train_img, train_pred, S, C, B, count='BofB', scale=True)
    #     pil_image = draw_image(train_img[0], pred_bbox[0], S, B, count='BofB')
    #     image = wandb.Image(pil_image, caption=f"Training Bounding Boxes {epoch}")
    #     train_bboxes.append(image)

    #     pred_bbox = choice_boxes(train_img, train_pred, S, C, B, count='1ofB', scale=True)
    #     pil_image = draw_image(train_img[0], pred_bbox[0], S, B=1, count='BofB')
    #     image = wandb.Image(pil_image, caption=f"Choice Bounding Box {epoch}")
    #     train_bbox.append(image)
    #     # ======================================== #

    # train_epoch_loss = train_epoch_loss / len(train_iter)
    # pred_bbox = choice_boxes(train_img, train_pred, S, C, B, count='1of1', scale=False)
    
    # # Validation
    # with torch.no_grad():
    #     val_epoch_loss = 0
    #     model.eval()
    #     for val_img, val_target in val_iter:
    #         val_img, val_target = val_img.to(device), val_target.to(device)

    #         val_pred = model(val_img)
    #         val_iter_loss = criterion(val_pred, val_target).detach()

    #         val_epoch_loss += val_iter_loss
    #     model.train()

    # print('time >> {:.4f}\tepoch >> {:04d}\ttrain_loss >> {:.4f}\\tval_loss >> {:.4f}'
    #       .format(time.time()-start_time, epoch, train_epoch_loss, val_epoch_loss))
    
    # if (epoch+1) % 5 == 0:
    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         }, log_dir / f'epoch({epoch})_loss({train_epoch_loss:.3f}).pt')
    # if USE_WANDB:
    #     wandb.log({'train_loss': train_epoch_loss,
    #                'val_loss': val_epoch_loss,
    #                'train_bboxes': train_bboxes,
    #                'train_bbox': train_bbox})

    # conv_targets = convert_scale(target, imgs, S)

    # for idx, (img, conv_target) in enumerate(zip(imgs, conv_targets)):
    #     img = plot_image(img, conv_target, S)
    #     img.save(f'./test_images/output_{idx}.png', 'png')

