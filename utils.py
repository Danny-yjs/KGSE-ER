import os
import sys
import json
import pickle
import random
import pandas as pd
import torch.nn as nn
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES']='0'

def pad_sequence(sequences, batch_first=False, padding_value=0, max_len = 95):
    r"""Pad a list of variable length Tensors with zero

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *`` where `T` is the
            length of the longest sequence.
        Function assumes trailing dimensions and type of all the Tensors
            in sequences are same.

    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if batch_first is False
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    # sequences = sequences.unsqueeze(1)# 在第二维度加1，代表batch
    max_size = sequences[0].size()
    # trailing_dims = max_size[1:]
    trailing_dims = max_size[1:]
    # max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[0:length, i, ...] = tensor

    return out_tensor


def get_videos():
    video_path = []  # 存储训练集的所有图片路径
    points_path = []
    label = []  # 存储训练集图片对应索引信息
    # for cla in os.listdir(path) 只遍历path第一层，不往里走
    # path = "/home/ubuntu/nas/FB-GER/FABO/"
    path = "/home/ubuntu/nas/FB-GER/FABO/"
    # path_json = "/home/ubuntu/nas/FB-GER/FABO_json/"
    path_json = "/home/ubuntu/nas/FB-GER/FABO_json/"
    # path_S001 = "/home/ubuntu/nas/FB-GER/FABO/S001/"
    path_S001 = "/home/ubuntu/nas/FB-GER/FABO/S001/"
    # path_points = root + "/" + for_what + "_points"
    emotion_class = [cla for cla in os.listdir(path_S001) if os.path.isdir(os.path.join(path_S001, cla))]
    # !!!排序，保证顺序一致
    emotion_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(emotion_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('train_class_indices.json', 'w') as train_json_file:
        train_json_file.write(json_str)

    every_class_num = {}  # 存储每个类别的样本总数
    # supported = [".jpg", ".JPG", ".png", ".PNG", ".json"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for S_number in os.listdir(path):  # S001,S002
        S_path = os.path.join(path, S_number)  # D:\datasets\FABO_bjb_test\S001
        path_points = os.path.join(path_json, S_number)
        for cla in os.listdir(S_path):   # emotion
            cla_path = os.path.join(S_path, cla)
            cla_path_points = os.path.join(path_points, cla)
            # 遍历获取supported支持的所有文件路径
            video = []
            points = []
            for i in os.listdir(cla_path):   # 24_1
                # if os.path.splitext(i)[-1] in supported:
                #     images.append(os.path.join(path, cla, i))
                #     i_json = os.path.splitext(i)[0] + ".json"
                #     points.append(os.path.join(path_points, cla, i_json))
                if os.listdir(os.path.join(cla_path, i)):
                    video.append(os.path.join(cla_path, i))
                    # i_json = os.path.splitext(i)[0] + ".json"
                    points.append(os.path.join(cla_path_points, i))
                    # label.append(image_class)
    
            # images = [os.path.join(path, cla, i) for i in os.listdir(cla_path)
            #                 if os.path.splitext(i)[-1] in supported]
            # points = [os.path.join(path_points, cla, i) for i in os.listdir(cla_path_points)
            #                 if os.path.splitext(i)[-1] in supported]
            # 获取该类别对应的索引
            video_class = class_indices[cla]
            # 记录该类别的样本数量
            if cla not in every_class_num:
                every_class_num[cla] = len(video)
            else:
                every_class_num[cla] = every_class_num[cla] + len(video)
            for vid_path in video:
                video_path.append(vid_path)
                label.append(video_class)
            for poi_path in points:
                points_path.append(poi_path)
    return video_path, label, points_path, every_class_num


def read_split_data(root: str, val_rate: float = 0.2014):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    video_path, label, points_path, every_class_num = get_videos()
    train_video_path, train_points_path, train_video_label, val_video_path, val_points_path, val_video_label = [], [], [], [], [], []
    val_class_num, train_class_num = {}, {}
    # =======================按比例随机采样验证样本=========================
    val_path = random.sample(video_path, k=int(len(video_path) * val_rate))

    for i in range(0, len(video_path)):
        vid_path = video_path[i]
        if video_path[i] in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
            val_video_path.append(video_path[i])
            val_points_path.append(points_path[i])
            val_video_label.append(label[i])
            if str(label[i]) not in val_class_num:
                val_class_num[str(label[i])] = 1
            else:
                val_class_num[str(label[i])] = val_class_num[str(label[i])] + 1
        else:  # 否则存入训练集
            train_video_path.append(vid_path)
            train_video_label.append(label[i])
            train_points_path.append(points_path[i])
            if str(label[i]) not in train_class_num:
                train_class_num[str(label[i])] = 1
            else:
                train_class_num[str(label[i])] = train_class_num[str(label[i])] + 1

    # =================================================================

    print("{} images for training,{} for each class".format(len(train_video_path), train_class_num))
    print("{} images for validation, {} for each class.".format(len(val_video_path), val_class_num))
    # print("{} images for test.".format(len(test_images_path)))

    return train_video_path, train_video_label, train_points_path, val_video_path, val_video_label, val_points_path


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


class LSoftmaxLoss(nn.Module):
    def __init__(self, num_classes=7, margin=4):
        super(LSoftmaxLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.theta = nn.Parameter(torch.FloatTensor(num_classes, num_classes-1))
        nn.init.kaiming_uniform_(self.theta)

    def forward(self, input, target):
        batch_size = input.size(0)
        input_norm = torch.norm(input, p=2, dim=1, keepdim=True)
        input_normalized = input.div(input_norm.expand_as(input))

        target_onehot = torch.zeros(batch_size, self.num_classes).to(input.device)
        target_onehot.scatter_(1, target.view(-1, 1), 1)
        target_onehot.sub_(input_normalized * (1 - self.margin))

        output = input_normalized.mm(self.theta)
        loss = nn.CrossEntropyLoss()(output, target)

        return loss


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    # l_loss_function = LSoftmaxLoss().to(device)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    # loss_images = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images = data['frames'].to(device)
        face = data['face_points'].to(device)
        body = data['points'].to(device)
        l_hand = data['lh_points'].to(device)
        r_hand = data['rh_points'].to(device)
        labels = data['Y'].to(device)
        # images, labels = data    # 这里MyDataset将表情、骨骼数据与标签一起压缩，这里直接获取多个值是否可行
        sample_num += images.shape[0]
        # pred = model((face.to(device), body.to(device), l_hand.to(device), r_hand.to(device)))
        # N_feature, pred = model(images)
        pred, top_face_so, top_body_so = model(images, face, body, l_hand, r_hand)
        pred_classes = torch.max(pred, dim=1)[1]
        labels = torch.tensor(labels, dtype=torch.int64)
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        # loss_images = 0
        # for n in range(0, N):
        #     pred_image = N_feature[:, n, :]
        #     loss_image = loss_function(pred_image.to(device), labels.to(device))
        #     loss_images += loss_image
        loss_two = 0
        for i in range(0, top_face_so.shape[0]):
            # print(top_face_so[i, :].device, labels.to(device).device)
            loss_1 = loss_function(top_face_so[i, :].cuda(), labels.to(device))*6
            # print(loss_1.device)
            loss_2 = loss_function(top_body_so[i, :].cuda(), labels.to(device))*4
            # print(loss_2.device)
            loss_two = loss_two + loss_1 + loss_2
            # print(loss_two.device)
        loss_all = loss_function(pred, labels.to(device))*10
        loss = loss_two + loss_all
        # loss = loss_images + loss_video
        loss.backward()
        accu_loss += loss.detach()  # 把loss取出来，相当于获取loss值

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    # l_loss_function = LSoftmaxLoss().to(device)
    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        images = data['frames'].to(device)
        face = data['face_points'].to(device)
        body = data['points'].to(device)
        l_hand = data['lh_points'].to(device)
        r_hand = data['rh_points'].to(device)
        labels = data['Y'].to(device)
        # images, labels = data    # 这里MyDataset将表情、骨骼数据与标签一起压缩，这里直接获取多个值是否可行
        sample_num += images.shape[0]

        pred, top_face_so, top_body_so = model(images, face, body, l_hand, r_hand)
        pred_classes = torch.max(pred, dim=1)[1]
        labels = torch.tensor(labels, dtype=torch.int64)
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss_two = 0
        for i in range(0, top_face_so.shape[0]):
            # print(top_face_so[i, :].device, labels.to(device).device)
            loss_1 = loss_function(top_face_so[i, :].cuda(), labels.to(device))*6
            # print(loss_1.device)
            loss_2 = loss_function(top_body_so[i, :].cuda(), labels.to(device))*4
            # print(loss_2.device)
            loss_two = loss_two + loss_1 + loss_2
            # print(loss_two.device)
        loss_all = loss_function(pred, labels.to(device))*10
        loss = loss_two + loss_all

        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        # data_loader.desc = "[test epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
        #                                                                       accu_loss.item() / (step + 1),
        #                                                                       accu_num.item() / sample_num)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
