import os
import math
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils
from my_dataset import MyDataSet
from torch.utils.data import Dataset, DataLoader
from model import vit_base_patch16_224_in21k as create_model  
from utils import read_split_data, train_one_epoch, evaluate
import pandas as pd
import xlwt
import xlrd
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.ion()

os.environ['CUDA_VISIBLE_DEVICES']='0'

import warnings
warnings.filterwarnings("ignore")


import csv

def read_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, face_point, point, lh_point, rh_point, label = sample['frame'], sample['face_point'], sample['point'], \
                                                              sample['left_hand_point'], sample['right_hand_point'], \
                                                              sample['Y']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))

        return {'frame': img, 'face_point': face_point, 'point': point,
                'left_hand_point': lh_point, 'right_hand_point': rh_point, 'Y': label}


class RandomCrop(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, face_point, point, lh_point, rh_point, label = sample['frame'], sample['face_point'], sample['point'], \
                                                              sample['left_hand_point'], sample['right_hand_point'], \
                                                              sample['Y']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h,
                left: left + new_w]
        # scaler = {}
        # feats = [face_point, point, lh_point, rh_point]
        # for x in feats:
        #     all_data = np.vstack(x)
        #     scaler[x] = MinMaxScaler()
        #     scaler[x].fit(all_data)
        #
        # rh_point = [scaler['hands_right'].transform(x) for x in rh_point]
        # lh_point = [scaler['hands_left'].transform(x) for x in lh_point]
        # point = [scaler['bodies'].transform(x) for x in point]
        # face_point = [scaler['face'].transform(x) for x in face_point]

        return {'frame': image, 'face_point': face_point, 'point': point,
                'left_hand_point': lh_point, 'right_hand_point': rh_point, 'Y': label}


class Center(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, face_point, point, lh_point, rh_point, label = sample['frame'], sample['face_point'], sample['point'], \
                                                              sample['left_hand_point'], sample['right_hand_point'], \
                                                              sample['Y']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = int((h - new_h + 1) * 0.5)
        left = int((w - new_w + 1) * 0.5)
        image = image[top: top + new_h,
                left: left + new_w]

        return {'frame': image, 'face_point': face_point, 'point': point,
                'left_hand_point': lh_point, 'right_hand_point': rh_point, 'Y': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors.
    """
    def __call__(self, sample):
        image, face_point, point, lh_point, rh_point, label = sample['frame'], sample['face_point'], sample['point'], \
                                                              sample['left_hand_point'], sample['right_hand_point'], \
                                                              sample['Y']

        image = image.transpose((2, 0, 1))
        return {'frame': torch.from_numpy(image),
                'face_point': torch.from_numpy(face_point),
                'point': torch.from_numpy(point),
                'left_hand_point': torch.from_numpy(lh_point),
                'right_hand_point': torch.from_numpy(rh_point),
                'Y': torch.from_numpy(np.array(label))
                }


def main(args):
    best_acc = 0

    device = 'cuda'

    train_videos_path, train_labels, train_points_path, val_videos_path, val_labels, val_points_path = \
        read_split_data(args.data_path)


    transformed_train_dataset = MyDataSet(videos_path=train_videos_path,
                                          videos_class=train_labels,
                                          points_path=train_points_path,
                                          transform=transforms.Compose([
                                                       Rescale(256),
                                                       RandomCrop(224),
                                                       ToTensor()
                                                   ]))

    transformed_val_dataset = MyDataSet(videos_path=val_videos_path,
                                        videos_class=val_labels,
                                        points_path=val_points_path,
                                        transform=transforms.Compose([
                                                       Rescale(256),
                                                       Center(224),
                                                       ToTensor()
                                                   ]))


    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    print('Using {} dataloader workers every process'.format(nw))

    train_dataloader = DataLoader(transformed_train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=nw,
                                  collate_fn=transformed_train_dataset.collate_fn)

    val_dataloader = DataLoader(transformed_val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=nw,
                                collate_fn=transformed_val_dataset.collate_fn)

    model_vit = create_model(num_classes=11, has_logits=False).to(device)

    if args.weights == "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    else:
        weights_dict = torch.load(args.weights, map_location=device)

        del_keys = ['_']
        for k in del_keys:
            del weights_dict[k]

    if args.freeze_layers:
        miss_list = model_vit.load_state_dict(weights_dict, strict=False)[0]
        for name, para in model_vit.named_parameters():
            if name not in miss_list:
                para.requires_grad_(True)
            else:
                para.requires_grad_(False)
                
    pg = [p for p in model_vit.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    df = pd.DataFrame(columns=['time', 'step', 'train Loss', 'training accuracy', 'val Loss', 'val accuracy'])  # 列名
    df.to_csv("./val_acc.csv", index=False)  

    for epoch in range(args.epochs):
        sheet2.write(epoch + 1, 0, epoch + 1)
        sheet2.write(epoch + 1, 5, str(optimizer.state_dict()['param_groups'][0]['lr']))

        train_loss, train_acc = train_one_epoch(model=model_vit,
                                                optimizer=optimizer,
                                                data_loader=train_dataloader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        sheet2.write(epoch + 1, 1, str(train_loss))
        sheet2.write(epoch + 1, 2, str(train_acc))

        val_loss, val_acc = evaluate(model=model_vit,
                                     data_loader=val_dataloader,
                                     device=device,
                                     epoch=epoch)

        sheet2.write(epoch + 1, 3, str(val_loss))
        sheet2.write(epoch + 1, 4, str(val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model_vit.state_dict(), "./_") 

        torch.save(model_vit.state_dict(), "./_")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--use_fusion', action="store_true", dest="use_fusion", default=False, help='use images and points fusion')
    parser.add_argument('--use_branch', action="store_true", dest="use_branch", default="images", help='use branch: images,face_points, body_points')
    parser.add_argument('--use_points_body', action="store_true", dest="use_points_body", default=False, help='use points body for dnn')
    parser.add_argument('--use_points_face', action="store_true", dest="use_points_face", default=False, help='use points face for dnn')

    parser.add_argument('--first_layer_size', type=int, default=768)
    parser.add_argument('--confidence_threshold', type=float, default=0.1)
    parser.add_argument('--top_N_frames', type=float, default=10)

    parser.add_argument('--data_path', type=str, default="/_")
    parser.add_argument('--model_name', default='', help='create model name')

    parser.add_argument('--weights', type=str, default='./_', help='initial weights path')

    parser.add_argument('--freeze_layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
