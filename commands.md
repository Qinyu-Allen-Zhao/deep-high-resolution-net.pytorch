#### train on poseXv060, test on poseXv060

python tools/test.py --cfg experiments/poseX/resnet/res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE output/poseXv060/pose_resnet/res50_256x192_d256x3_adam_lr1e-3/model_best.pth TEST.USE_GT_BBOX True



#### train on poseXv060, test on coco

python tools/test.py --cfg experiments/poseX/hrnet/test-on-coco_w32_256x192_adam_lr1e-3.yaml TEST.MODEL_FILE output/poseXv060/pose_hrnet/w32_256x192_adam_lr1e-3/model_best.pth TEST.USE_GT_BBOX True



#### train on coco, test on poseXv060

python tools/test.py --cfg experiments/poseX/resnet/test-on-poseV060_res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE models/pytorch/pose_coco/pose_resnet_50_256x192.pth TEST.USE_GT_BBOX True