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
import h5py

logger = logging.getLogger(__name__)


class H36MDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 17
        self.flip_pairs = [[3, 6], [2, 5], [1, 4], [11, 14], [12, 15], [13, 16]]
        self.parent_ids = [7, 0, 1, 2, 0, 4, 5, 8, 9, 10, 10, 8, 11, 12, 8, 14, 15]

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self.db = self._get_db()

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        file_name = os.path.join(
            self.root, 'annot', '%s_images.txt' % self.image_set
        )
        with open(file_name, 'r') as file:
            img_names = [l.strip() for l in file.readlines()]

        anno_name = os.path.join(
            self.root, 'annot', '%s.h5' % self.image_set
        )
        f = h5py.File(anno_name, 'r')

        gt_db = []
        for i in range(len(img_names)):
            image_name = img_names[i]

            c = np.array(f['center'][i], dtype=np.float)
            s = np.array([f['scale'][i], f['scale'][i]], dtype=np.float)

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float)
            if self.image_set != 'test':
                joints = np.array(f['part'][i])
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = 1
                joints_3d_vis[:, 1] = 1

            image_dir = 'images'
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
        f.close()

        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        gt_file = os.path.join(
            cfg.DATASET.TEST_ROOT,
            'annot',
            '%s.h5' % cfg.DATASET.TEST_SET
        )
        f = h5py.File(gt_file, 'r')
        pos_gt_src = np.array(f['part']).transpose([1, 2, 0])
        f.close()

        pos_pred_src = np.transpose(preds, [1, 2, 0])
        uv_error = pos_pred_src[:,0:2] - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)

        rmse = np.mean(uv_err, axis=1)

        name_value = [
            ('Head', rmse[10]),
            ('Shoulder', 0.5 * (rmse[11] + rmse[14])),
            ('Elbow', 0.5 * (rmse[12] + rmse[15])),
            ('Wrist', 0.5 * (rmse[13] + rmse[16])),
            ('Hip', 0.5 * (rmse[1] + rmse[4])),
            ('Knee', 0.5 * (rmse[2] + rmse[5])),
            ('Ankle', 0.5 * (rmse[3] + rmse[6])),
            ('Mean', np.mean(rmse))
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']
