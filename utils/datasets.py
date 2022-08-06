import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms

#-------------------------------------
#对数据进行填充函数（specific_picture_3）
#-------------------------------------

def   pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        #---------------------------------------------------
        #读取照片和label相对路径（specific_picture_1）
        #---------------------------------------------------
        #读取照片路径文件内容
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        #将读取照片文件的内容更改为label标签内容
        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    #---------------------------------------------------
    #将照片数据和标签读入（specific_picture_2）
    #---------------------------------------------------

    def __getitem__(self, index):

        # ---------------------
        #  Image and label path
        # ---------------------
        #将读入的图片路径一行一行读出
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        #组装照片路径形成绝对路径
        img_path = 'D:\\coco' + img_path
        
        #print (img_path)
        #按照图片读出名称拼接相对应的照片绝对路径
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        label_path = 'D:\\PyTorch-YOLOv3\\PyTorch-YOLOv3\\data\\coco\\labels' + label_path
        print(label_path)
        if os.path.exists(label_path):
            # ---------
            #  Image
            # ---------
            # Extract image as PyTorch tensor

            #读取图片文件，并转换成tensor格式（跟numpy差不多，但是专门用于GPU）
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

            #数据必须是3个通道的（红色通道，绿色通道，蓝色通道），RGB格式不会出现不是3个通道的
            if len(img.shape) != 3:
                img = img.unsqueeze(0)
                img = img.expand((3, img.shape[1:]))

            #对图像进行填充
            #将数据处理成等高，等宽
            #输出图片数组的深度+高+宽
            _, h, w = img.shape
            #将宽和高读取出来留着以后用
            h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
            # 通过pad_to_square将图片修改为正方形
            img, pad = pad_to_square(img, 0)
            _, padded_h, padded_w = img.shape

            # ---------
            #  Label
            # ---------

            #读取照片并修改因填充而导致的label变化
            
        if os.path.exists(label_path):
            #读取标签信息
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            #修改因为填充而改变的label信息
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h
            #在创建好的数组前面多填一列0，以后存储东西用
            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

            # 对图像进行增强
            if self.augment:
                if np.random.random() < 0.5:
                    img, targets = horisontal_flip(img, targets)
        else:
            targets,img,img_path = None,None,None
        return img_path, img, targets

    #---------------------------------------------------
    #将__getitem__传出的图片进行缩放（specific_picture_4）
    #---------------------------------------------------

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        #去掉为空的部分
        targets = [boxes for boxes in targets if boxes is not None]
        imgs    = [boxes for boxes in imgs    if boxes is not None]
        paths   = [boxes for boxes in paths    if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # 将照片随机缩放
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
