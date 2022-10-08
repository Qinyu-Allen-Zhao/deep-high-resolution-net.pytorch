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


class PosexMPIIDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 16

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        file_name = os.path.join(
            self.root, 'annotations', 'posex_total_annotations.pkl'
        )
        anno = joblib.load(file_name)
        image_dir = 'images/validation'
        gt_db = self.read_annotations(anno, image_dir)

        file_name = os.path.join(
            self.root, 'random', 'posex_random_annotations.pkl'
        )
        anno = joblib.load(file_name)
        image_dir = 'random/images'
        gt_db += self.read_annotations(anno, image_dir)

        return gt_db

    def read_annotations(self, anno, image_dir):
        gt_db = []
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=np.float)
            s = np.array(a['scale'], dtype=np.float)

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            if self.image_set != 'test':
                joints = np.array(a['joints']).transpose()
                joints[:, 0:2] = joints[:, 0:2] - 1
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = 1
                joints_3d_vis[:, 1] = 1

            gt_db.append(
                {
                    'image': os.path.join(self.root, image_dir, image_name),
                    'center': c,
                    'scale': s,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                }
            )
        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        return {'Null': 0.0}, 0.0
