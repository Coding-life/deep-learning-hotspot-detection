import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import cv2
import torch.utils.data as data
from tqdm import tqdm
from torchvision import transforms

cv2.setNumThreads(0)
# 原始图片大小为1110*1110，将其分块dct（10*10）
class BlockDctTransform:
    # 1100*1100 -> 10*10*36
    def __call__(self, x):
        x = np.array(x)
        # x = x / 255     # 像素值为0-1

        x = np.float32(x)

        # 将 RGB 图像转换为灰度图像
        if x.shape[-1] == 3:  # 如果有三个颜色通道
            x = np.mean(x, axis=-1, keepdims=False)  # 求三个通道的平均值

        # 分块
        blocks = []
        for i in range(10):
            for j in range(10):
                # 对图像块进行dct变换
                img_block = x[i*111:(i+1)*111, j*111:(j+1)*111]
                block_dct = cv2.dct(img_block)
                blocks.append(block_dct)

        # 从每个图像块的dct系数矩阵中取出最小的36个数
        results = []
        for block in blocks:
            flatten_block = np.ndarray.flatten(block)
            sorted_block = np.sort(np.abs(flatten_block))[0:36]
            results.append(sorted_block)

        # 将结果转换为10*10*36的元组
        results = np.array(results).reshape(10, 10, 36)
        return results

# img/hotspots/orig/001.png
# img/hotspots/variants/002.png

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.labels = []
        self.imgs = []
        self.transform = transform
        for label in os.listdir(root_dir):
            for dir in os.listdir(os.path.join(root_dir, label)):
                for file in os.listdir(os.path.join(root_dir, label, dir)):
                    if label == 'hotspots' and self.labels.count(0) >= 90000:
                        continue
                    if label == 'non_hotspots' and self.labels.count(1) >= 90000:
                        continue
                    self.labels.append(0 if label == 'hotspots' else 1)
                    self.imgs.append(os.path.join(root_dir, label, dir, file))



    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.labels[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.labels)

class MyDataset(Dataset):  # 继承Dataset类
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(label, dtype=torch.int64)
        if label.item() > 1:
            print(label.item())
        return img, label

    def __len__(self):
        return len(self.labels)  # 返回图片的长度

if __name__ == "__main__":
    # image_path_train = "./benchmark-litho-aug/train_img"
    image_path_train = "D:/Python/HotspotDetection/benchmark-litho-aug/test_img"
    trainset = CustomDataset(root_dir=image_path_train,
                                        transform=BlockDctTransform())
    batchsize = 1
    nw = 4
    trainloader = data.DataLoader(trainset, batchsize, shuffle=False, num_workers=nw, pin_memory=True)
    train_bar = tqdm(trainloader)  # 进度条
    dct_imgs = []
    labels = []
    for step, data in enumerate(train_bar):
        img, label = data
        img = np.array(img).reshape(-1)
        label = np.array(label).reshape(-1)
        dct_imgs.append(img)
        labels.append(label)
    # dct_imgs = np.vstack(dct_imgs)
    # labels = np.vstack(labels)
    np.savetxt('E:/val_dct.csv', np.array(dct_imgs).reshape(-1, 3600), delimiter=',')
    np.savetxt('E:/val_label.csv', np.array(labels).reshape(-1, 1), delimiter=',')