import os
import numpy as np
from PIL import Image
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
import modified_vgg
import importlib
importlib.reload(modified_vgg)

class CocoDataset(torch.utils.data.Dataset):

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
        class_id = next(self.sampler)
        while True:
            # !! no guarantees that we iterate all elements,
            # TODO: modify this iterate one by one? or shift to weighted random sampler? !!
            cwid = (instance_index + 1) % self.cw_num_instances[class_id]
            # is_mask is_crowd reee
            image, masks, is_crowd = self.load_image_gt(class_id, cwid)
            impulse, gt_response, is_bad_image = self.generate_targets(masks, class_id, is_crowd)
            print(is_crowd,is_bad_image,class_id)
            if not is_bad_image:
                # channels first
                print("asdfasdf")
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
                print("asdduejrieorieuorieuo")
                instance_index += 1

    def __len__(self):
        return sum(self.cw_num_instances)

    def weighted_sampler(self):
        config = self.config
        # TODO: define weighted sampler weights based on this
        data_order = config.DATA_ORDER
        class_weighting = np.array(self.cw_num_instances)**0.5
        # adjust number of bg instances reweighting
        class_weighting[0] = np.median(class_weighting)
        class_weighting = class_weighting / np.sum(class_weighting)
        np.random.seed()
        while True:
            yield np.random.choice(self.class_ids, p=class_weighting)

    def generate_targets(self, masks, class_id, is_crowd):
        num_classes = self.config.NUM_CLASSES
        mask = masks[:, :, 0]
        umask = masks[:, :, 1]
        # what other bad cases? add them here
        # currently crowd, small sized masks are flagged as bad instances
        print("sicourd:",is_crowd,"sied",np.sum(mask))
        if class_id < 0 or np.sum(mask) < 256 or is_crowd:
            return None, None, True
        if np.sum(umask) / np.sum(mask) < 0.3:
            umask = mask
        # currently impulses are produced to fine tune for classification.
        # in future impulse gen code needs to be written
        impulse = umask
        gt_response = mask
        return impulse, gt_response, False

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
        is_mask = instance_info['is_mask']
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
        return image, masks, is_mask


def get_loader(dataset_cid, config):
    coco_dataset = CocoDataset(dataset_cid, config)
    data_loader = torch.utils.data.DataLoader(dataset=coco_dataset,
                                              batch_size=config.BATCH_SIZE,
                                              # collate_fn=_collate_fn,
                                              shuffle=True,
                                              # pin_memory=True,
                                              num_workers=0)
    return data_loader


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

    def __init__(self,init_weights = True):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(576, 256, (3, 3), padding = (1, 1))
        self.gap = nn.AvgPool2d((7, 7), stride=1)
        self.fc = nn.Linear(256, 81)
        self.relu = nn.ReLU(inplace=True)
        if init_weights:
            nn.init.xavier_uniform_(self.conv1.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.gap(x)
        x = self.relu(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


class SimpleHGModel(nn.Module):

    def __init__(self):
        super(SimpleHGModel, self).__init__()
        self.vgg = modified_vgg.vgg11_features(vgg_weights=True)
        self.class_predictor = Classifier()
        # self.mask_predictor = MaskProp()

    def forward(self, x):
        x = torch.cat(x, dim=1)
        c, m = self.vgg(x)
        return self.class_predictor(c)  # ,self.mask_predictor(c,m)


def loss_criterion(pred_mask, gt_mask, pred_class, gt_class, class_weighting):
    # if gt_class[0] == 1:
    #     return classification_loss(pred_class,gt_class)
    # else:
    #     return mask_loss(pred_mask,gt_mask) + classification_loss(pred_class,gt_class)
    # return mask_loss(pred_mask, gt_mask) #+ classification_loss(pred_class, gt_class)
    return classification_loss(pred_class, gt_class, class_weighting)

# pred_mask: N,1,w,h
# gt_mask: N,1,w,h


def mask_loss(pred_mask, gt_mask):
    # need to modify this
    # mask_shape = pred_mask.shape[2:]
    # F.max_pool2d()
    fg_size = gt_mask.squeeze().sum(-1).sum(-1).view(-1, 1, 1, 1)
    bg_size = (1 - gt_mask).squeeze().sum(-1).sum(-1).view(-1, 1, 1, 1)
    # bgfg_weighting = (gt_mask== 1).float()/fg_size + (gt_mask == 0).float()/bg_size
    bgfg_weighting = gt_mask + (gt_mask == 0).float() * fg_size / bg_size
    _loss = nn.BCEWithLogitsLoss(weight=bgfg_weighting)
    return _loss(pred_mask, gt_mask)


def classification_loss(pred_class, gt_class, class_weighting):
    # _loss = nn.CrossEntropyLoss()
    _loss = nn.BCEWithLogitsLoss(weight=class_weighting)
    return _loss(pred_class, gt_class)


# TODO: modify dummy stub to train code or inference code
def main():
    return 0
if __name__ == '__main__':
    main()
