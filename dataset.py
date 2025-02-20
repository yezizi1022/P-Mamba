import os
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import custom_transforms as tr
import tifffile as tiff
import math
import glob
import tqdm

class RemoteData(data.Dataset):
    def __init__(self, base_dir='/mnt/data/data', mode="train", dataset='vaihingen', crop_szie=None, val_full_img=False):
        super(RemoteData, self).__init__()
        self.dataset_dir = base_dir
        self.mode = mode
        self.dataset = dataset
        self.val_full_img = val_full_img
        self.images = []
        self.labels = []
        self.names = []  # 文件名列表
        self.alphas = []
        if crop_szie is None:
            crop_szie = [512, 512]
        self.crop_size = crop_szie

        self.image_dir = os.path.join(self.dataset_dir, self.dataset, "images", f"{self.mode}")
        self.label_dir = os.path.join(self.dataset_dir, self.dataset, "annos", f"{self.mode}")

        imgs_path = glob.glob(os.path.join(self.image_dir, "*.[pj][pn]g"))
        for img_path in tqdm.tqdm(imgs_path):
            label_path = img_path.replace("images", "annos")
        # 下面是针对dynamic数据
        # for img_path in tqdm.tqdm(imgs_path):
        #     import re
        #     # 使用正则表达式替换路径中的 "images" 为 "annos"
        #     label_path = re.sub(r'images', 'annos', img_path)
        #     # 保留原始的扩展名
        #     label_path = label_path.replace('.jpg', '.png')
            ####
            assert os.path.exists(label_path), f"{label_path} must exist"
            self.names.append(os.path.basename(img_path))  # 存储文件名
            image = Image.open(img_path)
            image = np.array(image)
            label = Image.open(label_path)
            label = np.array(label) / 255.0
            self.images.append(image)
            self.labels.append(label)

        assert(len(self.images) == len(self.labels))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = {'image': self.images[index], 'label': self.labels[index]}
        sample = self.transform(sample)
        if self.val_full_img or self.mode != "train":  # 确保在验证/测试时返回文件名
            sample['name'] = self.names[index]
        if self.alphas != [] and self.mode != "train":
            sample['alpha'] = self.alphas[index]
        return sample

    def transform(self, sample):
        if self.mode == "train":
            composed_transforms = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.RandomVerticalFlip(),
                tr.RandomScaleCrop(base_size=self.crop_size, crop_size=self.crop_size),
                tr.RandomRotate(30),
                tr.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.1),  # 调整颜色扰动范围
                tr.GaussianBlur(kernel_size=(5, 5)),
                tr.ToTensor(add_edge=False),
            ])
        else:
            composed_transforms = transforms.Compose([
                tr.RandomScaleResize(base_size=self.crop_size),
                tr.ToTensor(add_edge=False),
            ])
        return composed_transforms(sample)


    def __str__(self):
        return 'dataset:{} mode:{}'.format(self.dataset, self.mode)

def label_to_RGB(image, classes=6):
    RGB = np.zeros(shape=[image.shape[0], image.shape[1], 3], dtype=np.uint8)
    if classes == 6:  # potsdam and vaihingen
        palette = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    if classes == 4:  # barley
        palette = [[255, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    for i in range(classes):
        index = image == i
        RGB[index] = np.array(palette[i])
    return RGB

def RGB_to_label(image=None, classes=6):
    if classes == 6:  # potsdam and vaihingen
        palette = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    if classes == 4:  # barley
        palette = [[255, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    label = np.zeros(shape=[image.shape[0], image.shape[1]], dtype=np.uint8)
    for i in range(len(palette)):
        index = image == np.array(palette[i])
        index[..., 0][index[..., 1] == False] = False
        index[..., 0][index[..., 2] == False] = False
        label[index[..., 0]] = i
    return label

