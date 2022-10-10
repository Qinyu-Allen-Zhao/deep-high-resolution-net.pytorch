# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mpii import MPIIDataset as mpii
from .coco import COCODataset as coco
from .poseX import PoseXDataset as poseX
from .posex_mpii import PosexMPIIDataset as posex_mpii
from .posex_h36m import PosexH36MDataset as posex_h36m
from .h36m import H36MDataset as h36m
from .posex_random import PosexRandomDataset as posex_ran
from .eva_ran_mpii import EvaRanOnMPII as eva_mpii
from .eva_ran_coco import EvaRanOnCOCO as eva_coco
from .posex_mpii_total import PosexMPIICombDataset as pose_mpii_comb
from .posex_pro import PosexProDataset as posex_pro