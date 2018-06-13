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

import skimage.measure

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
            loaded_im, loaded_masks = image, masks
            image, impulse, gt_response, one_hot, is_bad_image = self.generate_targets(image, masks, class_id, is_crowd)
            if not is_bad_image:
                image = image / 256
                image -= self.config.MEAN_PIXEL
                image /= self.config.STD_PIXEL
                # channels first
                image = np.moveaxis(image, 2, 0)
                # impulse = np.moveaxis(np.expand_dims(impulse, -1), 2, 0)
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
            impulse = impulse * 255
            response = np.squeeze(response) * 255
            Image.fromarray(image.astype(np.uint8), "RGB").show()
            for i in range(impulse.shape[0]):
                Image.fromarray(impulse[i].astype(np.uint8), "L").show()
            Image.fromarray(response.astype(np.uint8), "L").show()
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
        if m[0].shape == (0,):
            print(m)
        # y1,x1,y2,x2. bottom right just outside of blah
        return np.min(m[0]), np.min(m[1]), np.max(m[0]) + 1, np.max(m[1]) + 1

    def random_crop(self, image_obj, umask_obj, mask_obj, b, bbox):
        w, h = umask_obj.size
        y1, x1, y2, x2 = bbox
        p = int(random.uniform(max(0, x2 - b), min(w - b, x1)))
        q = int(random.uniform(max(0, y2 - b), min(h - b, y1)))
        return image_obj.crop((p, q, p + b, q + b)), umask_obj.crop((p, q, p + b, q + b)), mask_obj.crop((p, q, p + b, q + b))

    def random_impulse(self, umask_obj):
        umask = np.array(umask_obj)
        small_umask = skimage.measure.block_reduce(umask, (8, 8), np.min)
        w, h = umask.shape
        if not np.any(small_umask):
            y1, x1, y2, x2 = self.extract_bbox(umask)
            locx = (y1 + y2) // 2
            locy = (x1 + x2) // 2
        else:
            idx = np.where(small_umask)
            locx, locy = random.choice(list(zip(idx[0], idx[1])))
            locx, locy = locx * 8, locy * 8
            # y1, x1, y2, x2 = self.extract_bbox(umask)
            # locx = (y1 + y2) // 2
            # locy = (x1 + x2) // 2
        # dimensions of impulse
        l = 8
        n = 4
        impulse = np.zeros((3 * n - 2,) + umask.shape)
        impulse[0][max(locx - l // 2, 0):min(locx + l // 2, w), max(locy - l // 2, 0):min(locy + l // 2, h)] = 1
        for i in range(n - 1):
            L = (3**i) * l
            impulse[3 * i + 1][max(locx - L // 2, 0):min(locx + L // 2, w), max(locy - L, 0):min(locy + L, h)] = 1
            impulse[3 * i + 2][max(locx - L, 0):min(locx + L, w), max(locy - L // 2, 0):min(locy + L // 2, h)] = 1
            impulse[3 * i + 3][max(locx - L, 0):min(locx + L, w), max(locy - L, 0):min(locy + L, h)] = 1
        return impulse
    # unscaled image, masks

    def generate_targets(self, image, masks, class_id, is_crowd):

        config = self.config
        num_classes = self.config.NUM_CLASSES

        mask = masks[:, :, 0]
        umask = masks[:, :, 0]

        # very small objects. will be ignored now and retrained later
        # should probably keep crowds like oranges etc
        if is_crowd or np.sum(mask) < 50:
            return None, None, None, None, True

        # umask_obj now denotes where it is the only object present
        # mask_obj is the ground truth annotation
        # using umask_obj we generate impulse
        image_obj = Image.fromarray(image, "RGB")
        mask_obj = Image.fromarray(mask, "L")
        umask_obj = Image.fromarray(umask, "L")

        image_obj = self.resize_image(image_obj, (672, 672), "RGB")
        mask_obj = self.resize_image(mask_obj, (672, 672), "L")
        umask_obj = self.resize_image(umask_obj, (672, 672), "L")

        # if np.sum(np.array(umask_obj)) / np.sum(np.array(mask_obj)) < 0.3:
        #     umask = mask
        # code to crop stuff y1, x1, y2, x2
        bbox = self.extract_bbox(np.array(mask_obj))
        y1, x1, y2, x2 = bbox
        b = config.CROP_SIZE

        # big object
        if (x2 - x1) > 180 or (y2 - y1) > 180:
            image_obj = self.resize_image(image_obj, (b, b), "RGB")
            umask_obj = self.resize_image(umask_obj, (b, b), "L")
            mask_obj = self.resize_image(mask_obj, (b, b), "L")
        # small object
        else:
            image_obj, umask_obj, mask_obj = self.random_crop(image_obj, umask_obj, mask_obj, b, bbox)

        if np.sum(np.array(umask_obj)) < 20:
            return None, None, None, None, True
        impulse = self.random_impulse(umask_obj)
        impulse[-1] = np.array(mask_obj)
        gt_response = mask_obj
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
                                              num_workers=config.NUM_WORKERS
                                              )
    return data_loader


# proposes masks for single scale feature maps.
# for multi_scale supervision, use this multiple times


class MaskProp(nn.Module):

    def __init__(self, init_weights=True):
        super(MaskProp, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2)
        self.layer5 = nn.Sequential(
            nn.Conv2d(640 + 128, 256, (3, 3), padding=(1, 1)), nn.BatchNorm2d(256), self.relu,
        )
        self.mask_layer5 = nn.Sequential(
            nn.Conv2d(256, 64, (3, 3), padding=(1, 1)), nn.BatchNorm2d(64), self.relu,
            nn.Conv2d(64, 10, (3, 3), padding=(1, 1)), nn.BatchNorm2d(10), self.relu,
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256 + 128, 256, (3, 3), padding=(1, 1)), nn.BatchNorm2d(256), self.relu,
        )
        self.mask_layer4 = nn.Sequential(
            nn.Conv2d(256, 64, (3, 3), padding=(1, 1)), nn.BatchNorm2d(64), self.relu,
            nn.Conv2d(64, 10, (3, 3), padding=(1, 1)), nn.BatchNorm2d(10), self.relu,
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256 + 64, 128, (3, 3), padding=(1, 1)), nn.BatchNorm2d(128), self.relu,
            nn.Conv2d(128, 32, (3, 3), padding=(1, 1)), nn.BatchNorm2d(32), self.relu,
        )
        self.mask_layer3 = nn.Sequential(
            nn.Conv2d(32, 10, (3, 3), padding=(1, 1)), nn.BatchNorm2d(10), self.relu,
        )
        if init_weights:
            for name, child in self.named_children():
                if name[:-1] == 'layer' or 'mask_layer':
                    for gc in child.children():
                        if isinstance(gc, nn.Conv2d):
                            nn.init.xavier_uniform_(gc.weight)

    def forward(self, x):
        c, m = x
        masks = []
        c = F.upsample(c, scale_factor=2)
        l3, l4, l5 = m

        y = self.layer5(torch.cat([c, l5], 1))
        masks.append(self.mask_layer5(y))
        y = self.upsample(y)

        y = self.layer4(torch.cat([y, l4], 1))
        masks.append(self.mask_layer4(y))
        y = self.upsample(y)

        y = self.layer3(torch.cat([y, l3], 1))
        masks.append(self.mask_layer3(y))
        y = self.upsample(y)

        y = self.upsample(y)
        return y,masks


# classifier takes a single level features and classifies


class Classifier(nn.Module):

    def __init__(self, init_weights=True):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(640, 256, (3, 3), padding=(1, 1))
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
# def expand_impulse(base_impulse):
#     idx = base_impulse.nonzero()
#     i1,j1,i2,j2 = idx[:,0].min(),idx[:,1].min(),idx[:,0].max(),idx[:,1].max()
#     l1,l2,l3 = torch.cuda.FloatTensor(base_impulse.shape).fill_(0)


class SimpleHGModel(nn.Module):

    def __init__(self):
        super(SimpleHGModel, self).__init__()
        self.vgg0 = modified_vgg.split_vgg16_features(pre_trained_weights=False,d_in=10)
        self.vgg1 = modified_vgg.split_vgg16_features(pre_trained_weights=False,d_in=32)
        self.mp0 = MaskProp()
        self.mp1 = MaskProp()
        # self.mp2 = MaskProp()
        # self.mp3 = MaskProp()
        self.class_predictor = Classifier()

    def forward(self, x):
        im, base_impulse = x
        del x
        outs = []
        inp = torch.cat([im, base_impulse], dim=1)
        class_features, mask_features = self.vgg0(inp)
        y,m0 = self.mp0([class_features, mask_features])
        outs.append(m0)
        # gen_impulse = F.threshold(F.sigmoid(F.upsample(m0[-1], scale_factor=4)),0.5,0)

        # base_impulse = base_impulse[:,:-1,...]
        
        inp = torch.cat([im, y], dim=1)
        class_features, mask_features = self.vgg1(inp)
        y,m1 = self.mp0([class_features, mask_features])
        outs.append(m1)
        # gen_impulse = F.upsample(m1[-1], scale_factor=4)

        # inp = torch.cat([im, base_impulse,gen_impulse], dim=1)
        # class_features, mask_features = self.vgg(inp)
        # m2 = self.mp2([class_features, mask_features])
        # outs.append(m2)
        # gen_impulse = F.upsample(m2[-1], scale_factor=4)

        # inp = torch.cat([im, base_impulse,gen_impulse], dim=1)
        # class_features, mask_features = self.vgg(inp)
        # m3 = self.mp3([class_features, mask_features])
        # outs.append(m3)
        # gen_impulse = F.upsample(m3[-1], scale_factor=4)

        c = self.class_predictor(class_features)
        return c, outs
        # return c,m0

def loss_criterion(pred_class, gt_class, pred_masks, gt_mask,base_impulse):
    idx = gt_class[..., 0].nonzero()
    # mask_weights = torch.cuda.FloatTensor(gt_class.shape[0]).fill_(1)
    mask_weights = torch.ones(gt_class.shape[0]).float()
    mask_weights[idx] = 0
    loss1 = classification_loss(pred_class, gt_class)
    loss2 = 0
    for i in range(2):
        loss2 += multi_scale_mask_loss(pred_masks[i], gt_mask, mask_weights,base_impulse)
    return loss1, loss2


def multi_scale_mask_loss(pred_masks,gt_mask,mask_weights,base_impulse):
    pred_masks = reversed(pred_masks)
    loss = 0
    for i in range(3):
        scale_down = 2**(i+2)
        pred = next(pred_masks)
        loss += mask_loss(pred,gt_mask,mask_weights,scale_down,base_impulse)*scale_down/4
        # print(loss)
    return loss 
# gt_mask: N,1,w,h
# pred_masks: N,1,w/scale_down,h/scale_down
def mask_loss(pred_masks, gt_mask, mask_weights,scale_down,base_impulse):
    target = F.max_pool2d(gt_mask,(scale_down,scale_down),stride = scale_down)
    impulse = F.max_pool2d(base_impulse,(scale_down,scale_down),stride = scale_down)
    fg_size = gt_mask.squeeze().sum(-1).sum(-1).view(-1, 1, 1, 1)/(scale_down**2)
    bg_size = (1 - gt_mask).squeeze().sum(-1).sum(-1).view(-1, 1, 1, 1)/(scale_down**2)
    target = target*impulse
    if (fg_size > 0).sum() != gt_mask.shape[0] or (bg_size > 0).sum() != gt_mask.shape[0]:
        print((fg_size>0).sum(),"sdfsdfg\n",(bg_size>0).sum())
    mask_weights = mask_weights.view(-1, 1, 1, 1).repeat(1,10,1,1)
    # bgfg_weighting = (target == 1).float() / fg_size + (target == 0).float() / bg_size
    # bgfg_weighting = (target == 1).float() + (target == 0).float()
    bgfg_weighting = ((target == 1).float() / fg_size + (target == 0).float() / bg_size) * (fg_size + bg_size)
    bgfg_weighting *= mask_weights
    _loss = nn.BCEWithLogitsLoss(weight=bgfg_weighting, reduce=False)
    l = _loss(pred_masks, target)
    # l = l.squeeze().sum(-1).sum(-1)
    l = l.mean()
    return l


def classification_loss(pred_class, gt_class):
    _loss = nn.BCEWithLogitsLoss(reduce=False)
    l = _loss(pred_class, gt_class)
    # l = l.sum(-1)
    l = l.mean()
    return l

# TODO: modify dummy stub to train code or inference code


def main():
    return 0
if __name__ == '__main__':
    main()
