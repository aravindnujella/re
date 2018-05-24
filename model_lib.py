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


class CocoDataset_cw_ins(torch.utils.data.Dataset):

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.cw_num_instances = [len(c) for c in dataset.class_wise_instance_info]
        self.class_ids = range(config.NUM_CLASSES)
        # class_wise_iterators
        self.sampler = self.weighted_sampler()
    # regardless of instance_index, we give some shit.
    # shouldn't matter anyway because of blah blah

    def __getitem__(self, instance_index):
        # skipping bad images. is this bad?
        # caution: cw dataset didnt note if its a crowd. needs fixing
        while True:
            class_id = next(self.sampler)
            # !! no guarantees that we iterate all elements !!
            cwid = (instance_index + 1) % self.cw_num_instances[class_id]
            image, masks, class_id = self.load_image_gt(class_id, cwid)
            impulse, gt_response, class_id, is_bad_image = self.generate_targets(masks, class_id)
            if not is_bad_image:
                # channels first
                image = image.astype(np.float32)
                image -= self.config.MEAN_PIXEL
                image = image / 128
                image = np.moveaxis(image, 2, 0)
                one_hot = np.zeros(81).astype(np.float32)
                one_hot[class_id] = 1
                impulse = np.moveaxis(np.expand_dims(impulse, -1), 2, 0).astype(np.float32)
                gt_response = np.moveaxis(np.expand_dims(gt_response, -1), 2, 0).astype(np.float32)
                return torch.from_numpy(image), torch.from_numpy(impulse), torch.from_numpy(gt_response), torch.tensor(
                    one_hot)
            else:
                instance_index += 1

    def __len__(self):
        return len(self.dataset.instance_info)

    def weighted_sampler(self):
        #     # TODO: repeatedly take am between median and freq
        class_weighting = np.array(self.cw_num_instances)**0.5
        # adjust number of bg instances reweighting
        class_weighting[0] = np.median(class_weighting)
        class_weighting = class_weighting / np.sum(class_weighting)
        np.random.seed()
        while True:
            yield np.random.choice(self.class_ids, p=class_weighting)
    # def set_sampler_weights(self):
    #     # TODO: repeatedly take am between median and freq
    #     class_weighting = np.array(self.cw_num_instances)**0.5
    #     # adjust number of bg instances reweighting
    #     class_weighting[0] = np.median(class_weighting)
    #     class_weighting = class_weighting / np.sum(class_weighting)
    #     self.class_weighting = class_weighting

    def generate_targets(self, masks, class_id):
        num_classes = self.config.NUM_CLASSES
        mask = masks[:, :, 0]
        umask = masks[:, :, 1]
        # what other bad cases? add them here
        # currently crowd, small sized masks are flagged as bad instances
        if class_id <= 0 or np.sum(mask) < 256:
            return None, None, None, True
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
        return image

    def load_image_gt(self, class_id, cwid):
        config = self.config
        dataset = self.dataset
        instance_info = dataset.class_wise_instance_info[class_id][cwid]
        image_path = instance_info["image_path"]
        mask_obj = instance_info["mask_obj"]
        # class_id = instance_info["class_id"]

        image = self.read_image(image_path)
        masks = maskUtils.decode(mask_obj)

        image, window, scale, padding = self.resize_image(
            image,
            min_dim=config.WIDTH,
            max_dim=config.HEIGHT,
            padding=config.IS_PADDED)
        masks = self.resize_mask(masks, scale, padding)
        if random.random() > 0.5:
            image = np.fliplr(image)
            masks = np.fliplr(masks)
        return image, masks, class_id


class CocoDataset_ins(torch.utils.data.Dataset):

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.cw_num_instances = [len(c) for c in dataset.class_wise_instance_info]
        self.class_ids = range(config.NUM_CLASSES)

    # regardless of instance_index, we give some shit.
    # shouldn't matter anyway because of blah blah
    def __getitem__(self, instance_index):
        # skipping bad images. is this bad?
        while True:
            image, masks, class_id = self.load_image_gt(instance_index)
            impulse, gt_response, class_id, is_bad_image = self.generate_targets(masks, class_id)
            if not is_bad_image:
                # channels first
                image = image.astype(np.float32)
                image -= self.config.MEAN_PIXEL
                image = image / 128
                image = np.moveaxis(image, 2, 0)
                one_hot = np.zeros(81).astype(np.float32)
                one_hot[class_id] = 1
                impulse = np.moveaxis(np.expand_dims(impulse, -1), 2, 0).astype(np.float32)
                gt_response = np.moveaxis(np.expand_dims(gt_response, -1), 2, 0).astype(np.float32)
                return torch.from_numpy(image), torch.from_numpy(impulse), torch.from_numpy(gt_response), torch.tensor(
                    one_hot)
            else:
                instance_index += 1

    def __len__(self):
        return len(self.dataset.instance_info)

    def generate_targets(self, masks, class_id):
        num_classes = self.config.NUM_CLASSES
        mask = masks[:, :, 0]
        umask = masks[:, :, 1]
        # what other bad cases? add them here
        # currently crowd, small sized masks are flagged as bad instances
        if class_id <= 0 or np.sum(mask) < 256:
            return None, None, None, True
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
        return image

    def load_image_gt(self, instance_index):
        config = self.config
        dataset = self.dataset
        instance_info = dataset.instance_info[instance_index]
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
        if random.random() > 0.5:
            image = np.fliplr(image)
            masks = np.fliplr(masks)
        return image, masks, class_id


# write custom collate to delete bad instances?
# not doing this because some times we might get all bad images
# batch = list of (images,impulse,gt_response,class_id)
# def _collate_fn(batch):
#     batch = filter(lambda x: x is not None, batch)
#     return torch.utils.data.dataloader.default_collate(list(batch))
# we take cid object from main (ugly interface)


def get_loader(dataset_cid, config, data_order="ins"):
    coco_dataset = None
    if data_order == "cw_ins":
        coco_dataset = CocoDataset_cw_ins(dataset_cid, config)
    else:
        coco_dataset = CocoDataset_ins(dataset_cid, config)
    data_loader = torch.utils.data.DataLoader(dataset=coco_dataset,
                                              batch_size=config.BATCH_SIZE,
                                              # collate_fn=_collate_fn,
                                              shuffle=True,
                                              num_workers=16)
    return data_loader

# 1) convolve 2) batchnorm 3) add residual


class BasicBlock(nn.Module):

    def __init__(self, in_planes, mid_planes, out_planes, fr=(3, 3)):
        # in_planes are mapped to out_planes by a bottleneck like connection,
        # mid_planes only get 3,3 kernel size convs -> used to control no. of parms
        # if more feature mixing is needed, use more of these
        super(BasicBlock, self).__init__()
        pad = ((fr[0] - 1) // 2, (fr[1] - 1) // 2)
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=(1, 1))
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=fr, padding=pad)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=(1, 1))
        self.bn3 = nn.BatchNorm2d(out_planes)
        # highway is a direct linear transform on input to match output dimensions
        self.highway = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.highway(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x += residual
        x = self.relu(x)

        return x

# proposes masks for single scale feature maps.
# for multi_scale super vision, use this multiple times


class MaskProp(nn.Module):

    def __init__(self, n_dims):
        super(MaskProp, self).__init__()
        self.bb1 = BasicBlock(n_dims, 32, 16)
        self.bb2 = BasicBlock(16, 8, 8)
        self.conv1 = nn.Conv2d(8, 1, (1, 1))

    def forward(self, x):
        x = self.bb1(x)
        x = self.bb2(x)
        x = self.conv1(x)
        return x

# classifier takes a single level features and classifies


class Classifier(nn.Module):

    def __init__(self, n_dims):
        super(Classifier, self).__init__()
        self.bb1 = BasicBlock(n_dims, 64, 64)
        self.gap = nn.AvgPool2d((7, 7), stride=1)
        self.fc = nn.Linear(64, 81)

    def forward(self, x):
        x = self.bb1(x)
        # x = F.max_pool2d(x, (2, 2), 2)
        x = self.gap(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        return x


# this is one down_sample and one up_sample
class SimpleHGModel(nn.Module):

    def __init__(self):
        super(SimpleHGModel, self).__init__()
        self.inp_conv_0 = nn.Conv2d(4, 16, (7, 7), padding=(3, 3))
        # output dimensions
        down_filter_sizes = [16, 32, 64, 64, 64, 64]
        bottleneck_filter_sizes = [16, 32, 64, 64, 64, 64]
        # wing_filter_sizes = [None,None,16,32,256,256]
        up_filter_sizes = [64, 64, 64, 64]

        self.down_conv_6 = BasicBlock(16, 64, down_filter_sizes[-6])

        self.down_conv_5 = BasicBlock(down_filter_sizes[-6], 64, down_filter_sizes[-5])

        self.down_conv_4 = BasicBlock(down_filter_sizes[-5], 64, down_filter_sizes[-4])

        self.down_conv_3 = BasicBlock(down_filter_sizes[-4], 64, down_filter_sizes[-3])

        self.down_conv_2 = BasicBlock(down_filter_sizes[-3], 64, down_filter_sizes[-2])

        self.down_conv_1 = BasicBlock(down_filter_sizes[-2], 64, down_filter_sizes[-1])

        # self.mid_conv_0 = BasicBlock(down_filter_sizes[-1], 64, up_filter_sizes[0])

        # self.up_conv_1 = BasicBlock(up_filter_sizes[0], 64, up_filter_sizes[1])

        # self.up_conv_2 = BasicBlock(up_filter_sizes[1], 64, up_filter_sizes[2])

        # self.up_conv_3 = BasicBlock(up_filter_sizes[2], 64, up_filter_sizes[3])

        # self.up_conv_4 = BasicBlock(up_filter_sizes[3], 64, up_filter_sizes[4])

        # self.up_conv_5 = BasicBlock(up_filter_sizes[4], 64, up_filter_sizes[5])

        # self.up_conv_6 = BasicBlock(up_filter_sizes[5], 64, up_filter_sizes[6])

        self.wing_conv_1 = BasicBlock(down_filter_sizes[-1], 64, up_filter_sizes[0])

        # self.wing_conv_2 = BasicBlock(down_filter_sizes[-2], 64, up_filter_sizes[1])

        # self.wing_conv_3 = BasicBlock(down_filter_sizes[-3], 64, up_filter_sizes[2])

        # self.wing_conv_4 = BasicBlock(down_filter_sizes[-4], 64, up_filter_sizes[3])

        # self.wing_conv_5 = BasicBlock(down_filter_sizes[-5], 64, up_filter_sizes[4])

        # self.wing_conv_6 = BasicBlock(down_filter_sizes[-6], 64, up_filter_sizes[5])

        # self.mask_predictor = MaskProp(up_filter_sizes[3])
        self.class_predictor = Classifier(up_filter_sizes[0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # elif isinstance(m, nn.BatchNorm2d):
        #     nn.init.constant_(m.weight, 128)
        #     nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # HourGlass
        image = x[0]
        impulse = x[1]
        inp = torch.cat([image, impulse], dim=1)
        # 6,6,4 -> 6,6,8
        inp = self.inp_conv_0(inp)
        wing_convs = []
        # 6,6,8->6,6,16->5,5,16
        inp = self.down_conv_6(inp);  # wing_convs.append(self.wing_conv_6(inp));
        inp = F.max_pool2d(inp, (2, 2), 2)
        # 5,5,16->5,5,32->4,4,32
        inp = self.down_conv_5(inp);  # wing_convs.append(self.wing_conv_5(inp));
        inp = F.max_pool2d(inp, (2, 2), 2)
        # 4,4,32->4,4,64->3,3,64
        inp = self.down_conv_4(inp);  # wing_convs.append(self.wing_conv_4(inp));
        inp = F.max_pool2d(inp, (2, 2), 2)
        # 3,3,64->3,3,128->2,2,128
        inp = self.down_conv_3(inp);  # wing_convs.append(self.wing_conv_3(inp));
        inp = F.max_pool2d(inp, (2, 2), 2)
        # 2,2,128->2,2,256->1,1,256
        inp = self.down_conv_2(inp);  # wing_convs.append(self.wing_conv_2(inp));
        inp = F.max_pool2d(inp, (2, 2), 2)
        # 1,1,256->1,1,512->0,0,512
        inp = self.down_conv_1(inp); wing_convs.append(self.wing_conv_1(inp));
        # inp = F.max_pool2d(inp, (2, 2), 2)
        # 0,0,512->0,0,512
        # inp = self.mid_conv_0(inp);
        # # up_convs = []
        # # 1,1,256<-1,1,512<-0,0,512
        # inp = F.upsample(inp, scale_factor=2); inp = self.up_conv_1(inp + wing_convs[-1]); # up_convs.append(inp)
        # # 2,2,128<-2,2,256<-1,1,256
        # inp = F.upsample(inp, scale_factor=2); inp = self.up_conv_2(inp + wing_convs[-2]); # up_convs.append(inp)
        # # 3,3,64<-3,3,128<-2,2,128
        # inp = F.upsample(inp, scale_factor=2); inp = self.up_conv_3(inp + wing_convs[-3]); # up_convs.append(inp)
        # 4,4,32<-4,4,64<-3,3,64
        # inp = F.upsample(inp, scale_factor=2); inp = self.up_conv_4(inp + wing_convs[-4]); # up_convs.append(inp)
        # # 5,5,16<-5,5,32<-4,4,32
        # inp = F.upsample(inp, scale_factor=2); inp = self.up_conv_5(inp + wing_convs[-5]); # up_convs.append(inp)
        # # 6,6,8<-6,6,16<-5,5,16
        # inp = F.upsample(inp, scale_factor=2); inp = self.up_conv_6(inp + wing_convs[-6]); # up_convs.append(inp)

        # Maskprediction, classification
        return self.class_predictor(wing_convs[-1])  # F.upsample(self.mask_predictor(inp),scale_factor = 8)


def loss_criterion(pred_mask, gt_mask, pred_class, gt_class,class_weighting):
    # if gt_class[0] == 1:
    #     return classification_loss(pred_class,gt_class)
    # else:
    #     return mask_loss(pred_mask,gt_mask) + classification_loss(pred_class,gt_class)
    # return mask_loss(pred_mask, gt_mask) #+ classification_loss(pred_class, gt_class)
    # return classification_loss(pred_class, gt_class, class_weighting)

# pred_mask: N,1,w,h
# gt_mask: N,1,w,h
def mask_loss(pred_mask, gt_mask):
    # need to modify this
    # mask_shape = pred_mask.shape[2:]
    # F.max_pool2d()
    fg_size = gt_mask.squeeze().sum(-1).sum(-1).view(-1,1,1,1)
    bg_size = (1-gt_mask).squeeze().sum(-1).sum(-1).view(-1,1,1,1)
    # bgfg_weighting = (gt_mask== 1).float()/fg_size + (gt_mask == 0).float()/bg_size
    bgfg_weighting = gt_mask + (gt_mask == 0).float()*fg_size/bg_size
    _loss = nn.BCEWithLogitsLoss(weight=bgfg_weighting)
    return _loss(pred_mask, gt_mask)


def classification_loss(pred_class, gt_class, class_weighting):
    # _loss = nn.CrossEntropyLoss()
    _loss = nn.BCEWithLogitsLoss(weight=class_weighting)
    return 81*_loss(pred_class, gt_class)


# TODO: modify dummy stub to train code or inference code
def main():
    return 0
if __name__ == '__main__':
    main()
