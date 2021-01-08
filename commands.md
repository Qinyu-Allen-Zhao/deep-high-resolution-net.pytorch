#### train on poseXv070-occ1, test on poseXv070-occ1

python tools/test.py --evalExcludeKpt 0 --cfg experiments/poseX/resnet/v070-occ1/res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE output/v070-occ1/poseXv070-occ1/pose_resnet/res50_256x192_d256x3_adam_lr1e-3/model_best.pth TEST.USE_GT_BBOX True



#### train on poseXv070, test on poseXv070-occ1

...



### train on poseXv070-occ1, test on poseXv070

python tools/test.py --evalExcludeKpt 0 --cfg experiments/poseX/resnet/v070-occ1/test-on-v070_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE output/v070-occ1/poseXv070-occ1/pose_resnet/res50_256x192_d256x3_adam_lr1e-3/model_best.pth TEST.USE_GT_BBOX True



---

#### train on coco, test on coco

+ 17: python tools/test.py --evalExcludeKpt 0 --cfg experiments/poseX/resnet/v070/test-on-coco_res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE models/pytorch/pose_coco/pose_resnet_50_256x192.pth  TEST.USE_GT_BBOX False



#### train on poseXv070, test on poseXv070

python tools/test.py --evalExcludeKpt 0 --cfg experiments/poseX/resnet/v070/res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE output/v070/poseXv070/pose_resnet/res50_256x192_d256x3_adam_lr1e-3/model_best.pth TEST.USE_GT_BBOX True



#### train on coco, test on poseXv070

+ 12: python tools/test.py --evalExcludeKpt 5 --cfg experiments/poseX/resnet/v070/test-on-v070_res50_256x192_d256x3_adam_lr1e-3.yaml  TEST.MODEL_FILE models/pytorch/pose_coco/pose_resnet_50_256x192.pth  TEST.USE_GT_BBOX True

  

#### train on poseXv070, test on coco

+ 12: python tools/test.py --evalExcludeKpt 5 --cfg experiments/poseX/resnet/v070/test-on-coco_res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE output/v070/poseXv070/pose_resnet/res50_256x192_d256x3_adam_lr1e-3/model_best.pth TEST.USE_GT_BBOX True



### train on mpii, vali on mpii

+ train: python tools/train.py --cfg experiments/poseX/resnet/v070/mpii_256x192_d256x3_adam_lr1e-3.yaml 

---

#### train on poseXv060, test on poseXv060

python tools/test.py --evalExcludeKpt 0 --cfg experiments/poseX/resnet/v060/res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE output/v060/poseXv060/pose_resnet/res50_256x192_d256x3_adam_lr1e-3/model_best.pth TEST.USE_GT_BBOX True



#### train on poseXv060, test on coco

+ 12: python tools/test.py --evalExcludeKpt 5 --cfg experiments/poseX/resnet/v060/test-on-coco_res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE output/v060/poseXv060/pose_resnet/res50_256x192_d256x3_adam_lr1e-3/model_best.pth TEST.USE_GT_BBOX True



#### train on coco, test on poseXv060

+ 12: python tools/test.py --evalExcludeKpt 5 --cfg experiments/poseX/resnet/v060/test-on-v060_res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE models/pytorch/pose_coco/pose_resnet_50_256x192.pth TEST.USE_GT_BBOX True



#### train on poseXv060-occ1(test on poseXv060-occ1)

+ train: python tools/train.py  --cfg experiments/poseX/resnet/v060/res50_256x192_d256x3_adam_lr1e-3.yaml 
+ test: python tools/test.py --evalExcludeKpt 0 --cfg experiments/poseX/resnet/v060/res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE output/poseXv060-occ1/pose_resnet/res50_256x192_d256x3_adam_lr1e-3/model_best.pth TEST.USE_GT_BBOX True



#### train on poseXv060, test on poseXv060-occ1

python tools/test.py --evalExcludeKpt 0 --cfg experiments/poseX/resnet/v060/test-on-060-occ1_res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE output/poseXv060/pose_resnet/res50_256x192_d256x3_adam_lr1e-3/model_best.pth TEST.USE_GT_BBOX True



#### train on poseXv060-occ1, test on poseXv060

python tools/test.py --evalExcludeKpt 0 --cfg experiments/poseX/resnet/v060/test-on-poseV060_res50_256x192_d256x3_adam_lr1e-3.yaml TEST.MODEL_FILE output/poseXv060-occ1/pose_resnet/res50_256x192_d256x3_adam_lr1e-3/model_best.pth TEST.USE_GT_BBOX True

