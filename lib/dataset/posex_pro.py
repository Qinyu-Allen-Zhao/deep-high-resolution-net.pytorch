# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json_tricks as json
from collections import OrderedDict

import numpy as np
from scipy.io import loadmat, savemat

from dataset.JointsDataset import JointsDataset
import joblib


logger = logging.getLogger(__name__)


class PosexProDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 16

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        gt_db = []
        i = 0

        img_path = os.path.join(self.root, 'images')
        ann_path = os.path.join(self.root, 'ann')

        for sub_id in range(103):
            ann = joblib.load(os.path.join(ann_path, 'sub%d.pkl' % sub_id))
            center = ann['center'].T
            scale = ann['scale']
            all_joints = ann['keypoints']

            for img_id in range(500):
                image_file = os.path.join(img_path, "sub{}_{:0>8}.png".format(sub_id, img_id))

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
                        'scale': scale[img_id],
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
