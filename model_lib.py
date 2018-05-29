import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import random
import numpy as np
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import modified_vgg
import importlib
importlib.reload(modified_vgg)


class CocoDataset(torch.utils.data.Dataset):

    def __init__(self, cwid, config, data_dir):
        self.cwid = cwid
        self.config = config
        self.data_dir = data_dir
        self.cw_num_instances = [len(c) for c in cwid]
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
            image, masks, is_crowd = self.load_image_gt(class_id, cwid)
            image, impulse, gt_response, one_hot, is_bad_image = self.generate_targets(image, masks, class_id, is_crowd)
            if not is_bad_image:
                image = image/256
                image -= self.config.MEAN_PIXEL
                image /= self.config.STD_PIXEL 
                # channels first
                image = np.moveaxis(image, 2, 0)
                impulse = np.moveaxis(np.expand_dims(impulse, -1), 2, 0)
                gt_response = np.moveaxis(np.expand_dims(gt_response, -1), 2, 0)
                return torch.from_numpy(image), torch.from_numpy(impulse), torch.from_numpy(gt_response), torch.tensor(
                    one_hot)
            else:
                instance_index += 1

    def __len__(self):
        return sum(self.cw_num_instances)

    def visualize_data(self):
        n = len(self)
        for i in range(n):
            image, impulse, response, one_hot = [data.numpy() for data in self[i]]
            image = np.moveaxis(image,0,-1)
            image *= self.config.STD_PIXEL
            image += self.config.MEAN_PIXEL
            image *= 255
            image[:,:,0][np.where(impulse.squeeze()==1)] = 255
            impulse = np.squeeze(impulse)*255
            response = np.squeeze(response)*255
            Image.fromarray(image.astype(np.uint8),"RGB").show()
            # Image.fromarray(impulse.astype(np.uint8),"L").show()
            # Image.fromarray(response.astype(np.uint8),"L").show()
            print(self.config.CLASS_NAMES[np.argmax(one_hot)])
            input()

    def weighted_sampler(self):
        config = self.config
        # TODO: define weighted sampler weights based on data_order
        data_order = config.DATA_ORDER
        class_weighting = np.array(self.cw_num_instances)
        # class_weighting = class_weighting**0.5
        # class_weighting = np.log2(class_weighting)
        # adjust number of bg instances reweighting
        class_weighting[0] = np.median(class_weighting)
        print(class_weighting)
        class_weighting = class_weighting / np.sum(class_weighting)
        np.random.seed()
        while True:
            yield np.random.choice(self.class_ids, p=class_weighting)

    def extract_bbox(self, mask):
        m = np.where(mask != 0)
        # x1,y1,x2,y2
        return np.min(m[0]), np.min(m[1]), np.max(m[0]), np.max(m[1])

    def random_crop(self, image_obj, mask_obj, b, bbox):
        w, h = mask_obj.size
        y1, x1, y2, x2 = bbox
        p = int(random.uniform(max(0, x2 - b), min(w - 1 - b, x1)))
        q = int(random.uniform(max(0, y2 - b), min(h - 1 - b, y1)))
        return image_obj.crop((p, q, p + b, q + b)), mask_obj.crop((p, q, p + b, q + b))

    # unscaled image, masks
    def generate_targets(self, image, masks, class_id, is_crowd):
        config = self.config
        num_classes = self.config.NUM_CLASSES

        mask = masks[:, :, 0]
        umask = masks[:, :, 1]

        # very small objects. will be ignored now and retrained later
        # should probably keep crowds like oranges etc
        if is_crowd or np.sum(mask) < 50:
            return None, None, None, None, True
        
        if np.sum(umask) / np.sum(mask) < 0.3:
            umask = mask

        image_obj = Image.fromarray(image, "RGB")
        mask_obj = Image.fromarray(mask, "L")
        umask_obj = Image.fromarray(umask, "L")

        image_obj = self.resize_image(image_obj,(640,640),"RGB")
        mask_obj = self.resize_image(mask_obj,(640,640),"L")
        umask_obj = self.resize_image(umask_obj,(640,640),"L")

        # code to crop stuff y1, x1, y2, x2
        bbox = self.extract_bbox(np.array(umask_obj))
        y1, x1, y2, x2 = bbox
        b = config.CROP_SIZE
        
        # big object
        if (x2 - x1) > 100 or (y2 - y1) > 100:
            image_obj = self.resize_image(image_obj, (b, b), "RGB")
            umask_obj = self.resize_image(umask_obj, (b, b), "L")
        # small object
        else:
            image_obj, umask_obj = self.random_crop(image_obj, umask_obj, b, bbox)
        # currently impulses are produced to fine tune for classification.
        # in future impulse gen code needs to be written
        impulse = umask_obj
        gt_response = mask_obj
        one_hot = np.zeros(81)
        one_hot[class_id] = 1
        return np.array(image_obj).astype(np.float32), np.array(impulse).astype(np.float32), np.array(gt_response).astype(np.float32), np.array(one_hot).astype(np.float32), False

    def read_image(self, image_id):
        image = Image.open(self.data_dir + image_id).convert("RGB")
        return np.array(image)

    def resize_image(self, image_obj, max_dim, mode):
        z = Image.new(mode, max_dim, "black")
        image_obj.thumbnail(max_dim,Image.ANTIALIAS)
        (w, h) = image_obj.size
        z.paste(image_obj, ((max_dim[0] - w) // 2, (max_dim[1] - h) // 2))
        return z

    def load_image_gt(self, class_id, instance_index):
        config = self.config
        cwid = self.cwid
        instance_info = cwid[class_id][instance_index]
        image_id = instance_info["image_id"]
        mask_obj = instance_info["mask_obj"]
        is_crowd = instance_info['is_crowd']
        image = self.read_image(image_id)
        masks = maskUtils.decode(mask_obj)

        # if random.random() > 0.5:
        #     image = np.fliplr(image)
        #     masks = np.fliplr(masks)

        return image, masks, is_crowd


def get_loader(cwid, config, data_dir):
    coco_dataset = CocoDataset(cwid, config, data_dir)
    data_loader = torch.utils.data.DataLoader(dataset=coco_dataset,
                                              batch_size=config.BATCH_SIZE,
                                              # collate_fn=_collate_fn,
                                              shuffle=True,
                                              pin_memory=config.PIN_MEMORY,
                                              num_workers=config.NUM_WORKERS)
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

    def __init__(self, init_weights=True):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(576, 256, (3, 3), padding=(1, 1))
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
        self.vgg = modified_vgg.vgg11_features(pre_trained_weights=True)
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
