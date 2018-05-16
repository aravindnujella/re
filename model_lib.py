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
            instance_index += 1
        return torch.from_numpy(image), torch.from_numpy(impulse), torch.from_numpy(gt_response), torch.from_numpy(class_id)

    def __len__(self):
        return len(self.dataset.instance_info)

    def generate_targets(self, masks, class_id):
        num_classes = self.config.NUM_CLASSES
        mask = masks[:, :, 0]
        umask = masks[:, :, 1]
        # what other bad cases? add them here
        # currently crowd, zero sized masks are flagged as bad instances
        if class_id < 0 or np.sum(mask) == 0:
            return None,None,None,True
        # not bad image
        one_hot_class = np.zeros((num_classes,))
        one_hot_class[class_id] = 1
        # happens when cake on dining table, tie on human etc
        if np.sum(umask)/np.sum(mask) < 0.3:
            umask = mask
        # currently impulses are produced to fine tune for classification.
        # in future impulse gen code needs to be written
        impulse = umask
        gt_response = mask
        return impulse, gt_response, one_hot_class, False

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
