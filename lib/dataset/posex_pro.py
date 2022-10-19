# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import os
import random

import cv2
from glob import glob

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

from dataset.JointsDataset import JointsDataset
import joblib

logger = logging.getLogger(__name__)


class PosexProDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 16

        self.db = self._get_db()
        path = "/datasets/haven/output/"
        dirs = os.listdir(path)
        self.object_list = []
        for dir_name in dirs:
            if dir_name == '.ipynb_checkpoints':
                continue
            for file in range(5):
                self.object_list.append(os.path.join(path, dir_name, f"{file}.hdf5"))

        self.human_list = list(glob('/datasets/synthesis_dataset/syn_humans/*_img.pt'))
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])
        self.obj_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize(300),
            # Data augmentation
            transforms.RandomHorizontalFlip(0.5),
            transforms.Pad(padding=300, fill=0),
            transforms.RandomCrop(size=(512, 512))
        ])
        self.hum_transform = transforms.Compose([
            transforms.Resize(300),
            # Data augmentation
            transforms.RandomHorizontalFlip(0.5),
            transforms.Pad(padding=300, fill=0),
            transforms.RandomCrop(size=(512, 512))
        ])

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        gt_db = []
        i = 0

        img_path = os.path.join(self.root, 'images')
        ann_path = os.path.join(self.root, 'ann')

        for sub_id in range(10):
            file_sub = os.path.join(ann_path, 'sub%d.pkl' % sub_id)
            if not os.path.exists(file_sub):
                continue
            ann = joblib.load(file_sub)
            center = ann['center'].T
            scale = ann['scale']
            all_joints = ann['keypoints']

            for img_id in range(50):
                image_file = os.path.join(img_path, "sub{}_{:0>8}.png".format(sub_id, img_id))
                if not os.path.exists(image_file):
                    continue

                joints = all_joints[img_id]

                joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = 1
                joints_3d_vis[:, 1] = 1

                gt_db.append(
                    {
                        'image': image_file,
                        'center': center[img_id],
                        'scale': np.array([scale[img_id], scale[img_id]]),
                        'joints_3d': joints_3d,
                        'joints_3d_vis': joints_3d_vis,
                        'filename': image_file,
                        'imgnum': i,
                    }
                )
                i += 1

        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        return {'Null': 0.0}, 0.0

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                    and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        img = self.img_transform(input).permute((1, 2, 0))

        num_obj = np.random.randint(3)
        for _ in range(num_obj):
            choice = np.random.randint(0, len(self.object_list))
            f = h5py.File(self.object_list[choice], 'r')
            obj = np.array(f['colors']) / 255
            depth = np.array(f['depth'])
            obj[np.where(depth > 1e8)] = 0

            obj = self.obj_transform(obj).permute((1, 2, 0))
            img[np.where(depth < 1e8)] = 0
            img += obj

        num_hum = np.random.randint(2)
        for _ in range(num_hum):
            choice = np.random.randint(0, len(self.human_list))
            tsf_img = torch.load(self.human_list[choice], map_location=torch.device('cpu'))
            tsf_mask = torch.load(self.human_list[choice][:-7] + '_mask.pt', map_location=torch.device('cpu'))
            tsf_img = self.hum_transform(tsf_img)[0].permute((1, 2, 0))

            img = tsf_mask * img + (1 - tsf_mask) * (tsf_img + 1) / 2.0

        # if self.transform:
        #     input = self.transform(input)
        img = img.permute((2, 0, 1))
        input = normalize(img)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input, target, target_weight, meta
