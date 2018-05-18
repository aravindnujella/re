import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import scipy.misc
import skimage.color
import skimage.io
import skimage.measure
import random
import numpy as np
import time
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


class CocoDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config

    def __getitem__(self, instance_index):
        impulse, gt_response, class_id, is_bad_image = None, None, None, True
        # skipping bad images. is this bad?
        while is_bad_image:
            image, masks, class_id = self.load_image_gt(instance_index)
            impulse, gt_response, class_id, is_bad_image = self.generate_targets(
                masks, class_id)
            if is_bad_image:
                instance_index += 1
                continue
            else:
                # channels first
                image = np.moveaxis(image, 2, 0).astype(np.float32)
                impulse = np.moveaxis(np.expand_dims(impulse, -1), 2, 0).astype(np.float32)
                gt_response = np.moveaxis(np.expand_dims(gt_response, -1), 2, 0).astype(np.float32)
                break
        # print(class_id,"hey")
        return torch.from_numpy(image), torch.from_numpy(impulse), torch.from_numpy(gt_response), torch.tensor(
            class_id,dtype=torch.long)

    def __len__(self):
        return len(self.dataset.instance_info)

    def generate_targets(self, masks, class_id):
        num_classes = self.config.NUM_CLASSES
        mask = masks[:, :, 0]
        umask = masks[:, :, 1]
        # what other bad cases? add them here
        # currently crowd, zero sized masks are flagged as bad instances
        if class_id < 0 or np.sum(mask) == 0:
            return None, None, None, True
        # not bad image
        # one_hot_class = np.zeros((num_classes,)).astype(np.int64)
        # one_hot_class[class_id] = 1
        # happens when cake on dining table, tie on human etc
        if np.sum(umask) / np.sum(mask) < 0.3:
            umask = mask
        # currently impulses are produced to fine tune for classification.
        # in future impulse gen code needs to be written
        impulse = umask
        gt_response = mask
        return impulse, gt_response, class_id, False

    def resize_image(self, image, min_dim=None, max_dim=None, padding=False):
        # Default window (y1, x1, y2, x2) and default scale == 1.
        h, w = image.shape[:2]
        window = (0, 0, h, w)
        scale = 1

        # Scale?
        if min_dim:
            # Scale up but not down
            scale = max(1, min_dim / min(h, w))
        # Does it exceed max dim?
        if max_dim:
            image_max = max(h, w)
            if round(image_max * scale) > max_dim:
                scale = max_dim / image_max
        # Resize image and mask
        if scale != 1:
            image = scipy.misc.imresize(
                image, (round(h * scale), round(w * scale)))
        # Need padding?
        if padding:
            # Get new height and width
            h, w = image.shape[:2]
            top_pad = (max_dim - h) // 2
            bottom_pad = max_dim - h - top_pad
            left_pad = (max_dim - w) // 2
            right_pad = max_dim - w - left_pad
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0)
            window = (top_pad, left_pad, h + top_pad, w + left_pad)
        return image, window, scale, padding

    def resize_mask(self, mask, scale, padding):
        h, w = mask.shape[:2]
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
        return mask

    def read_image(self, image_path):
        image = skimage.io.imread(image_path)
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image.astype(np.float32)

    def load_image_gt(self, image_id):
        config = self.config
        dataset = self.dataset
        instance_info = dataset.instance_info[image_id]
        image_path = instance_info["image_path"]
        mask_obj = instance_info["mask_obj"]
        class_id = instance_info["class_id"]

        image = self.read_image(image_path)
        masks = maskUtils.decode(mask_obj)

        image, window, scale, padding = self.resize_image(
            image,
            min_dim=config.WIDTH,
            max_dim=config.HEIGHT,
            padding=config.IS_PADDED)
        masks = self.resize_mask(masks, scale, padding)
        # if random.random() > 0.5:
        #     image = np.fliplr(image)
        #     masks = np.fliplr(masks)
        return image, masks, class_id


# write custom collate to delete bad instances?
# def _collate_fn(batch):
#     batch = filter(lambda x: x is not None, batch)
#     return default_collate(batch)

# we take cid object from main (ugly interface)


def get_loader(dataset_cid, config):
    coco_dataset = CocoDataset(dataset_cid, config)
    data_loader = torch.utils.data.DataLoader(dataset=coco_dataset,
                                              batch_size=config.BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=8)
    return data_loader

# 1) convolve 2) batchnorm 3) add residual


class BasicBlock(nn.Module):

    def __init__(self, channel_sizes, expand_channels=0):
        super(BasicBlock, self).__init__()
        l = len(channel_sizes)
        layers = []
        for i in range(l - 1):
            layers.append(nn.Conv2d(
                in_channels=channel_sizes[i], out_channels=channel_sizes[i + 1], kernel_size=(3, 3), padding=1))
            layers.append(nn.BatchNorm2d(num_features=channel_sizes[i + 1], track_running_stats=True))
            layers.append(nn.ReLU())
        self.conv_block = nn.Sequential(*layers)
        # expand_block: expands to channel_sizes[-1] by default.
        # o.w. expands to specified number of channels using (1,1) convolutions
        if expand_channels == 0:
            expand_channels = channel_sizes[-1]
        self.expand_block = nn.Sequential(nn.Conv2d(channel_sizes[-1], expand_channels, kernel_size=(1, 1)),
                                          nn.BatchNorm2d(num_features=expand_channels, track_running_stats=True),
                                          nn.ReLU())

    def forward(self, x):
        x = F.relu(x + self.conv_block(x))
        x = self.expand_block(x)
        return x

# proposes masks for single scale feature maps.
# for multi_scale super vision, use this multiple times


class MaskProp(nn.Module):

    def __init__(self):
        super(MaskProp, self).__init__()
        self.bb1 = BasicBlock([8, 8, 8])
        self.conv1 = nn.Conv2d(8, 1, (1, 1))

    def forward(self, x):
        x = self.bb1(x)
        x = self.conv1(x)
        return x

# classifier takes a single level features and classifies


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.bb1 = BasicBlock([512, 512, 512],512)
        self.gap = nn.AvgPool2d((10, 10), stride=1)
        self.fc = nn.Linear(512, 81)

    def forward(self, x):
        x = self.bb1(x)
        x = F.max_pool2d(x, (2, 2), 2)
        x = self.gap(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


# this is one down_sample and one up_sample
class SimpleHGModel(nn.Module):

    def __init__(self):
        super(SimpleHGModel, self).__init__()
        self.inp_conv_0 = nn.Conv2d(4, 8, (7, 7), padding=3)

        self.down_conv_6 = BasicBlock([8, 8, 8], 16)

        self.down_conv_5 = BasicBlock([16, 16, 16], 32)

        self.down_conv_4 = BasicBlock([32, 32, 32], 64)

        self.down_conv_3 = BasicBlock([64, 64, 64], 128)

        self.down_conv_2 = BasicBlock([128, 128, 128], 256)

        self.down_conv_1 = BasicBlock([256, 256, 256], 512)

        self.mid_conv_0 = BasicBlock([512, 512, 512], 512)

        self.up_conv_1 = BasicBlock([512, 512, 512], 256)

        self.up_conv_2 = BasicBlock([256, 256, 256], 128)

        self.up_conv_3 = BasicBlock([128, 128, 128], 64)

        self.up_conv_4 = BasicBlock([64, 64, 64], 32)

        self.up_conv_5 = BasicBlock([32, 32, 32], 16)

        self.up_conv_6 = BasicBlock([16, 16, 16], 8)

        self.mask_predictor = MaskProp()
        self.class_predictor = Classifier()

    def forward(self, x):
        # HourGlass
        image = x[0]
        impulse = x[1]
        inp = torch.cat([image, impulse], dim=1)
        # 6,6,4 -> 6,6,8
        inp = self.inp_conv_0(inp)
        down_convs = []
        # 6,6,8->6,6,16->5,5,16
        inp = self.down_conv_6(inp); down_convs.append(inp); inp = F.max_pool2d(inp, (2, 2), 2)
        # 5,5,16->5,5,32->4,4,32
        inp = self.down_conv_5(inp); down_convs.append(inp); inp = F.max_pool2d(inp, (2, 2), 2)
        # 4,4,32->4,4,64->3,3,64
        inp = self.down_conv_4(inp); down_convs.append(inp); inp = F.max_pool2d(inp, (2, 2), 2)
        # 3,3,64->3,3,128->2,2,128
        inp = self.down_conv_3(inp); down_convs.append(inp); inp = F.max_pool2d(inp, (2, 2), 2)
        # 2,2,128->2,2,256->1,1,256
        inp = self.down_conv_2(inp); down_convs.append(inp); inp = F.max_pool2d(inp, (2, 2), 2)
        # 1,1,256->1,1,512->0,0,512
        inp = self.down_conv_1(inp); down_convs.append(inp); inp = F.max_pool2d(inp, (2, 2), 2)
        # 0,0,512->0,0,512
        inp = self.mid_conv_0(inp);
        # up_convs = []
        # 1,1,256<-1,1,512<-0,0,512
        inp = F.upsample(inp, scale_factor=2); inp = self.up_conv_1(inp + down_convs[-1]); # up_convs.append(inp)
        # 2,2,128<-2,2,256<-1,1,256
        inp = F.upsample(inp, scale_factor=2); inp = self.up_conv_2(inp + down_convs[-2]); # up_convs.append(inp)
        # 3,3,64<-3,3,128<-2,2,128        
        inp = F.upsample(inp, scale_factor=2); inp = self.up_conv_3(inp + down_convs[-3]); # up_convs.append(inp)
        # 4,4,32<-4,4,64<-3,3,64
        inp = F.upsample(inp, scale_factor=2); inp = self.up_conv_4(inp + down_convs[-4]); # up_convs.append(inp)
        # 5,5,16<-5,5,32<-4,4,32
        inp = F.upsample(inp, scale_factor=2); inp = self.up_conv_5(inp + down_convs[-5]); # up_convs.append(inp)
        # 6,6,8<-6,6,16<-5,5,16
        inp = F.upsample(inp, scale_factor=2); inp = self.up_conv_6(inp + down_convs[-6]); # up_convs.append(inp)

        # Maskprediction, classification
        return self.class_predictor(down_convs[-1]), self.mask_predictor(inp)


def loss_criterion(gt_mask, pred_mask, gt_class, pred_class):
    # if gt_class[0] == 1:
    #     return classfication_loss(gt_class,pred_class)
    # else:
    #     return mask_loss(gt_mask,pred_mask) + classification_loss(gt_class,pred_class)
    return mask_loss(gt_mask, pred_mask) + classification_loss(gt_class, pred_class)


def mask_loss(gt_mask, pred_mask):
    # need to modify this
    _loss = nn.SoftMarginLoss(reduce=True)
    return _loss(pred_mask, gt_mask)


def classification_loss(gt_class, pred_class):
    _loss = nn.CrossEntropyLoss()
    return _loss(pred_class, gt_class)
