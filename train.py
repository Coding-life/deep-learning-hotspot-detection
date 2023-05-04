import json
import os
import torch
import torch.utils.data as data
from torch import nn, optim
from torchvision import datasets, transforms
from tqdm import tqdm
from model import net
import numpy as np
import cv2
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
from custom_dataset import CustomDataset, BlockDctTransform, MyDataset
import csv

def read_dct(root):
    dct_img = []
    with open(root, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        pbar = tqdm(total=sum(1 for _ in reader))
        f.seek(0)  # 重新定位文件指针到文件开头
        for row in reader:
            dct_img.append(np.array(row, dtype=np.float32))
            pbar.update(1)
    pbar.close()
    dct_img = np.array(dct_img)
    dct_img = dct_img.reshape(-1, 10, 10, 36)
    return dct_img

def read_label(root):
    labels = []
    with open(root, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        pbar = tqdm(total=sum(1 for _ in reader))
        f.seek(0)  # 重新定位文件指针到文件开头
        for row in reader:
            labels.append(np.array(row, dtype=np.float64))
            pbar.update(1)
    pbar.close()
    labels = np.array(labels)
    labels = labels.reshape(-1)
    return labels


if __name__ == "__main__":
    num_devices = torch.cuda.device_count()
    # 打印所有可用设备的名称
    for i in range(num_devices):
        print(torch.cuda.get_device_name(i))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("Running on the device of '{}'.".format(device))


    image_path_train = "./benchmark-litho-aug/train_img"
    image_path_test = "./benchmark-litho-aug/test_img"
    # assert os.path.exists(image_path_train), "file '{}' does not exists.".format(image_path_train)
    # assert os.path.exists(image_path_test), "file '{}' does not exists.".format(image_path_test)
    # trainset = CustomDataset(root_dir=image_path_train,
    #                                 transform=transform["train"])

    train_dct = read_dct('dct_img.csv')
    train_label = read_label('label.csv')
    val_dct = read_dct('E:/val_dct.csv')
    val_label = read_label('E:/val_label.csv')

    # 计算均值和标准差
    # train_mean = np.mean(train_dct)
    # train_std = np.std(train_dct)
    # val_mean = np.mean(val_dct)
    # val_std = np.std(val_dct)
    train_mean = 2.8721884e-06
    train_std = 8.395721e-06
    val_mean = 2.8639167e-06
    val_std = 8.387624e-06
    print(train_mean, ' ', train_std, ' ', val_mean, ' ', val_std, '\n')

    transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(train_mean, train_std)]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(val_mean, val_std)])
    }

    trainset = MyDataset(imgs=train_dct, labels=train_label, transform=transform["train"])
    train_num = len(trainset)
    valset = MyDataset(imgs=val_dct, labels=val_label, transform=transform["val"])
    val_num = len(valset)

    # 针对训练集中的类别 0(HS) 进行欠采样
    # 计算正负样本比例
    # targets = np.array(trainset.labels)
    # num_positive = (targets == 0).sum()
    # num_negative = (targets == 1 ).sum()
    # print('%d %d\n' % (num_positive, num_negative))
    # positive_weight = num_negative / (num_positive + num_negative)
    # negative_weight = num_positive / (num_positive + num_negative)
    # positive_idxs = torch.where(torch.tensor(trainset.labels) == 0)[0]
    # negative_idxs = torch.where(torch.tensor(trainset.labels) == 1)[0]
    # negative_sampled_idxs = np.random.choice(negative_idxs, size=num_positive, replace=False)
    # sampled_idxs = torch.cat((positive_idxs, torch.tensor(negative_sampled_idxs)))
    # sampler = data.sampler.SubsetRandomSampler(sampled_idxs)

    batchsize = 64

    # nw = min(os.cpu_count(), batchsize if batchsize > 1 else 0, 8)
    nw = 0
    print("using {} dataloader workers every process".format(nw))
    trainloader = data.DataLoader(trainset, batchsize, shuffle=True, num_workers=nw, pin_memory=True)
    valloader = data.DataLoader(valset, batch_size=batchsize, shuffle=False, num_workers=nw, pin_memory=True)
    print("using {} images for trainning, {} images for validation.".format(train_num, val_num))

    model = net.to(device)
    # weights_path = './best.pth'
    # model.load_state_dict(torch.load(weights_path))
    # model = net.cuda()

    class FocalLoss(nn.Module):
        def __init__(self, gamma=0  , alpha=None):
            super(FocalLoss, self).__init__()
            self.gamma = gamma
            self.alpha = alpha

        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** self.gamma * ce_loss

            if self.alpha is not None:
                focal_loss = self.alpha[targets.to(inputs.device)].to(inputs.device) * focal_loss.to(inputs.device)

            return focal_loss.mean()

    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(alpha=torch.tensor([positive_weight, negative_weight]))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, verbose=True)

    # 定义早停参数
    early_stopping_patience = 10  # 设置 early stopping patience 为10
    best_val_loss = float('inf')  # 初始化验证集损失为正无穷
    current_patience = 0  # 当前耐心值

    epoches = 30
    save_path = './best.pth'
    train_acc = 0.0
    best_acc = 0.0
    best_acc_hs = 0.0
    # optimizer = optim.SGD(model.parameters(), learningrate, momentum=0.9)
    train_steps = len(trainloader)
    for epoch in range(epoches):
        total = 0  # 没用到
        train_acc = 0.0
        runningloss = 0.0
        # runningcorrect = 0.0

        model.train()  # model.train()是model对象继承nn.Module的的一个方法，表示模型处于训练模式，使用dropout层
        train_bar = tqdm(trainloader)  # 进度条
        hs_acc = 0.0
        nhs_acc = 0.0
        hs_num = 0
        nhs_num = 0
        for step, data in enumerate(train_bar):
            img, label = data
            outputs = model(img.to(device))
            loss = criterion(outputs, label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            runningloss += loss.item()
            predicted = torch.max(outputs.data, dim=1)[1]   # 网络预测的类别
            train_acc += torch.eq(predicted, label.to(device)).sum().item()
            total += label.size(0)
            correct = predicted == label.to(device)
            hs_samples = label.to(device) == 0
            nhs_samples = label.to(device) == 1
            correct_hs_samples = correct & hs_samples
            correct_nhs_samples = correct & nhs_samples
            hs_acc += correct_hs_samples.sum().item()
            nhs_acc += correct_nhs_samples.sum().item()
            hs_num += hs_samples.sum().item()
            nhs_num += nhs_samples.sum().item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.4f} ".format(epoch + 1, epoches, loss)
        train_accurate = train_acc / train_num
        train_loss = runningloss / train_steps
        hs_accurate = hs_acc / hs_num
        nhs_accurate = nhs_acc / nhs_num
        print('\n[epoch %d] train_loss: %.4f train_accuracy: %.4f hs_accuray: %.4f nhs_accuray: %.4f' %
              (epoch + 1, train_loss, train_accurate, hs_accurate, nhs_accurate))
        torch.cuda.empty_cache()
        model.eval()  # model.eval()是model对象继承nn.Module的的一个方法，表示模型处于验证模式，弃用dropout层
        acc = 0.0
        hs_acc = 0.0
        val_loss = 0.0
        hs_num = 0
        nhs_acc = 0.0
        nhs_num = 0.0
        # 不通过验证集更新网络
        with torch.no_grad():
            val_bar = tqdm(valloader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                val_loss += criterion(outputs, val_labels.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                correct = predict_y == val_labels.to(device)
                hs_samples = val_labels.to(device) == 0
                nhs_samples = val_labels.to(device) == 1
                correct_hs_samples = correct & hs_samples
                correct_nhs_samples = correct & nhs_samples
                hs_acc += correct_hs_samples.sum().item()
                nhs_acc += correct_nhs_samples.sum().item()
                hs_num += hs_samples.sum().item()
                nhs_num += nhs_samples.sum().item()
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        # 判断是否触发早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            current_patience = 0  # 重置耐心值
        else:
            current_patience += 1  # 增加耐心值

        if current_patience == early_stopping_patience:
            print("Early stopping triggered! Stop training.")
            break

        # 更新学习率
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        # 对学习率加上最小值限制
        lr = max(lr, 0.00001)
        optimizer.param_groups[0]['lr'] = lr

        val_accurate = acc / val_num
        hs_accurate = hs_acc / hs_num
        nhs_accurate = nhs_acc / nhs_num
        val_loss = val_loss / len(valloader)
        print('\n[epoch %d] val_loss: %.4f val_accuracy: %.4f hs_accuracy: %.4f nhs_accuracy: %.4f' %
              (epoch + 1, val_loss, val_accurate, hs_accurate, nhs_accurate))

        # 保存训练效果最好的一层的训练权重
        if hs_accurate > best_acc:
            best_acc = hs_accurate
            torch.save(net.state_dict(), save_path)
            torch.save(model, './save')

    print("Finshed Training")

    # model = torch.load('./path')
    # checkpoint = model.state_dict()
