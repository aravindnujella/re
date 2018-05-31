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
            loaded_im,loaded_masks = image,masks
            image, impulse, gt_response, one_hot, is_bad_image = self.generate_targets(image, masks, class_id, is_crowd)
            if not is_bad_image:
                image = image / 256
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
            image = np.moveaxis(image, 0, -1)
            image *= self.config.STD_PIXEL
            image += self.config.MEAN_PIXEL
            image *= 255
            image[:, :, 0][np.where(impulse.squeeze() == 1)] = 255
            image[:, :, 1][np.where(impulse.squeeze() == 1)] = 255
            image[:, :, 2][np.where(impulse.squeeze() == 1)] = 255
            impulse = np.squeeze(impulse) * 255
            response = np.squeeze(response) * 255
            Image.fromarray(image.astype(np.uint8), "RGB").show()
            # Image.fromarray(impulse.astype(np.uint8),"L").show()
            # Image.fromarray(response.astype(np.uint8),"L").show()
            print(self.config.CLASS_NAMES[np.argmax(one_hot)])
            input()

    def weighted_sampler(self):
        config = self.config
        # TODO: define weighted sampler weights based on data_order
        data_order = config.DATA_ORDER
        class_weighting = np.array(self.cw_num_instances)
        # class_weighting = np.log2(class_weighting)
        class_weighting = class_weighting**0.5
        class_weighting = class_weighting / np.sum(class_weighting)
        np.random.seed()
        while True:
            yield np.random.choice(self.class_ids, p=class_weighting)

    def extract_bbox(self, mask):
        m = np.where(mask != 0)
        # y1,x1,y2,x2. bottom right just outside of blah 
        return np.min(m[0]), np.min(m[1]), np.max(m[0])+1, np.max(m[1])+1

    def random_crop(self, image_obj, mask_obj, b, bbox):
        w, h = mask_obj.size
        y1, x1, y2, x2 = bbox
        p = int(random.uniform(max(0, x2 - b), min(w - b, x1)))
        q = int(random.uniform(max(0, y2 - b), min(h - b, y1)))
        s = np.sum(np.array(mask_obj))
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

        image_obj = self.resize_image(image_obj, (672, 672), "RGB")
        mask_obj = self.resize_image(mask_obj, (672, 672), "L")
        umask_obj = self.resize_image(umask_obj, (672, 672), "L")

        # code to crop stuff y1, x1, y2, x2
        bbox = self.extract_bbox(np.array(umask_obj))
        y1, x1, y2, x2 = bbox
        b = config.CROP_SIZE

        # big object
        if (x2 - x1) > 180 or (y2 - y1) > 180:
            image_obj = self.resize_image(image_obj, (b, b), "RGB")
            umask_obj = self.resize_image(umask_obj, (b, b), "L")
        # small object
        else:
            image_obj, umask_obj = self.random_crop(image_obj, umask_obj, b, bbox)
        # currently impulses are produced to fine tune for classification.
        # in future impulse gen code needs to be written
        impulse = umask_obj
        gt_response = umask_obj
        one_hot = np.zeros(81)
        one_hot[class_id] = 1
        return np.array(image_obj).astype(np.float32), np.array(impulse).astype(np.float32), np.array(gt_response).astype(np.float32), np.array(one_hot).astype(np.float32), False

    def read_image(self, image_id):
        image = Image.open(self.data_dir + image_id).convert("RGB")
        return np.array(image)

    def resize_image(self, image_obj, thumbnail_shape, mode):
        z = Image.new(mode, thumbnail_shape, "black")
        if mode == 'RGB':
            image_obj.thumbnail(thumbnail_shape, Image.ANTIALIAS)
        else:
            image_obj.thumbnail(thumbnail_shape, Image.NEAREST)
        (w, h) = image_obj.size
        z.paste(image_obj, ((thumbnail_shape[0] - w) // 2, (thumbnail_shape[1] - h) // 2))
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

    def __init__(self,init_weights = True):
        super(MaskProp, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.layer5 = nn.Sequential(
            nn.Conv2d(576+64, 288, (3, 3), padding=(1, 1)), nn.BatchNorm2d(288), self.relu,
            nn.Upsample(scale_factor=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(288+64, 144, (3, 3), padding=(1, 1)), nn.BatchNorm2d(144), self.relu,
            nn.Upsample(scale_factor=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(144+32, 36, (3, 3), padding=(1, 1)), nn.BatchNorm2d(36), self.relu,
            nn.Upsample(scale_factor=2),
        )   
        self.layer_ = nn.Sequential(
            nn.Conv2d(36, 1, (3, 3),padding=(1,1)), nn.BatchNorm2d(1), self.relu,
        )
        if init_weights:
            for name,child in self.named_children():
                if name[:-1] == 'layer':
                    for gc in child.children():
                        if isinstance(gc,nn.Conv2d):
                            nn.init.xavier_uniform_(gc.weight)

    def forward(self, x):
        c,m = x
        c = F.upsample(c,scale_factor=2)
        l3,l4,l5 = m
        y = self.layer5(torch.cat([c,l5],1))
        y = self.layer4(torch.cat([y,l4],1))
        y = self.layer3(torch.cat([y,l3],1))
        y = self.layer_(y)
        return y

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
        self.vgg = modified_vgg.vgg11_features(pre_trained_weights=False)
        self.mask_predictor = MaskProp()
        self.class_predictor = Classifier()

    def forward(self, x):
        x = torch.cat(x, dim=1)
        c, m = self.vgg(x)
        return self.class_predictor(c), self.mask_predictor([c,m])


def loss_criterion(pred_class, gt_class, pred_mask, gt_mask):
    gt_mask = F.upsample(gt_mask,size = pred_mask.shape[2:],mode="bilinear",align_corners=False)
    idx = gt_class[..., 0].nonzero()
    mask_weights = torch.cuda.FloatTensor(gt_class.shape[0]).fill_(1)
    mask_weights[idx] = 0
    loss1 = classification_loss(pred_class, gt_class)
    loss2 = mask_loss(pred_mask, gt_mask, mask_weights)
    return loss1, loss2

# pred_mask: N,1,w,h
# gt_mask: N,1,w,h


def mask_loss(pred_mask, gt_mask, mask_weights):
    fg_size = gt_mask.squeeze().sum(-1).sum(-1).view(-1, 1, 1, 1)
    bg_size = (1 - gt_mask).squeeze().sum(-1).sum(-1).view(-1, 1, 1, 1)
    mask_weights = mask_weights.view(-1, 1, 1, 1)
    # bgfg_weighting = (gt_mask == 1).float() / fg_size + (gt_mask == 0).float() / bg_size
    bgfg_weighting = (gt_mask == 1).float() / fg_size + (gt_mask == 0).float() / bg_size
    bgfg_weighting *= mask_weights
    _loss = nn.BCEWithLogitsLoss(weight=bgfg_weighting,reduce = False)
    l = _loss(pred_mask, gt_mask)
    l = l.squeeze().sum(-1).sum(-1)
    l = l.mean()
    return l


def classification_loss(pred_class, gt_class):
    _loss = nn.BCEWithLogitsLoss(reduce = False)
    l = _loss(pred_class, gt_class)
    l = l.sum(-1).mean()/81
    return l

# TODO: modify dummy stub to train code or inference code
def main():
    return 0
if __name__ == '__main__':
    main()
