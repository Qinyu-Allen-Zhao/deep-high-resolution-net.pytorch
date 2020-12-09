#### train on poseXv060, test on poseXv060

python tools/test.py --evalExcludeKpt 0 --cfg experiments/poseX/resnet/res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE output/poseXv060/pose_resnet/res50_256x192_d256x3_adam_lr1e-3/model_best.pth TEST.USE_GT_BBOX True



#### train on poseXv060, test on coco

python tools/test.py --evalExcludeKpt 0 --cfg experiments/poseX/resnet/test-on-coco_res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE output/poseXv060/pose_resnet/res50_256x192_d256x3_adam_lr1e-3/model_best.pth TEST.USE_GT_BBOX True

+ 12: python tools/test.py --evalExcludeKpt 5 --cfg experiments/poseX/resnet/test-on-coco_res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE output/poseXv060/pose_resnet/res50_256x192_d256x3_adam_lr1e-3/model_best.pth TEST.USE_GT_BBOX True



#### train on coco, test on poseXv060

python tools/test.py --evalExcludeKpt 0 --cfg experiments/poseX/resnet/test-on-poseV060_res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE models/pytorch/pose_coco/pose_resnet_50_256x192.pth TEST.USE_GT_BBOX True

+ 15: python tools/test.py --evalExcludeKpt 5 --cfg experiments/poseX/resnet/test-on-poseV060_res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE models/pytorch/pose_coco/pose_resnet_50_256x192.pth TEST.USE_GT_BBOX True



#### train on poseXv060-occ1(test on poseXv060-occ1)

+ train: python tools/train.py  --cfg experiments/poseX/resnet/res50_256x192_d256x3_adam_lr1e-3.yaml 
+ test: python tools/test.py --evalExcludeKpt 0 --cfg experiments/poseX/resnet/res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE output/poseXv060-occ1/pose_resnet/res50_256x192_d256x3_adam_lr1e-3/model_best.pth TEST.USE_GT_BBOX True



#### train on poseXv060, test on poseXv060-occ1

python tools/test.py --evalExcludeKpt 0 --cfg experiments/poseX/resnet/test-on-poseV060_res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE output/poseXv060/pose_resnet/res50_256x192_d256x3_adam_lr1e-3/model_best.pth TEST.USE_GT_BBOX True



#### train on poseXv060-occ1, test on poseXv060

python tools/test.py --evalExcludeKpt 0 --cfg experiments/poseX/resnet/test-on-poseV060_res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE output/poseXv060-occ1/pose_resnet/res50_256x192_d256x3_adam_lr1e-3/model_best.pth TEST.USE_GT_BBOX True