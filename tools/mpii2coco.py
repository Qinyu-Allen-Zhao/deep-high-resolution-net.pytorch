from scipy.io import loadmat, savemat
from PIL import Image
import os
import os.path as osp
import numpy as np
import json
import shutil

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x
# required file: mpii_human_pose_v1_u12_1.mat

def check_empty(list,name):

    try:
        list[name]
    except ValueError:
        return True

    if len(list[name]) > 0:
        return False
    else:
        return True

val_imgs = dict() # len = 2729
def get_val_imgs():
    val_json_path = "data/mpii/annot/valid.json"
    with open(val_json_path, "r") as f:
        val_json = json.load(f)
        for entry in val_json:
            if entry["image"] not in val_imgs.keys():
                val_imgs[entry["image"]] = 0
            val_imgs[entry["image"]] += 1

def is_train(img_name):
    if not val_imgs:
        get_val_imgs()
    return img_name not in val_imgs.keys()

"""change this to train, valid, test ot obtain different data split"""
db_type = 'valid'
annot_file = loadmat('data/mpii/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1')['RELEASE']
save_path = 'mpiiINcoco/' + "person_keypoints_" + db_type + '.json'
cpy_des_path = 'mpiiINcoco/' + db_type
img_path = "data/mpii/images/train/"

joint_num = 16
img_num = len(annot_file['annolist'][0][0][0])

aid = 0
# coco = {'images': [], 'categories': [], 'annotations': []}

"""coco format"""
info = {
    "description": "MPII in coco keypoint annotation",
    "url": "None",
    "version": "mpii_human_pose_v1_u12_1",
    "year": 2020,
    "contributor": "Xinqi Zhu(convertor)",
    "date_created": "2020/12/19",
}
licenses = [
    {
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License"
    }
]
categories = [
    {
        "supercategory": "person",
        "id": 1,
        "name": "person",
        "keypoints": [
            "None", "pelvis", "throax","upper_neck", "head_top",
            "left_shoulder","right_shoulder","left_elbow","right_elbow",
            "left_wrist","right_wrist","left_hip","right_hip", # COCO: 2 hip points are annotated differently
            "left_knee","right_knee","left_ankle","right_ankle"
        ],
        "skeleton": [
            [16,14],[14,12],[17,15],[15,13],[12,13],[6,7], # COCO: [6,12],[7,13]
            [6,8],[7,9],[8,10],[9,11], # COCO:[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
            [1,2],[1,3],[3,4],[4,5],[5,6],[5,7],[5,12], [5,13], """TODO: change!!!"""
        ]
    }
]
coco = {
    "info": info,
    "licenses": licenses,
    "images": [],
    "annotations": [],
    "categories": [],
}

skip_cnt = 0
for img_id in tqdm(range(img_num)):
    img_name = str(annot_file['annolist'][0][0][0][img_id]['image'][0][0][0][0])
    if db_type=="train" and not is_train(img_name):
        skip_cnt+=1
        continue
    if db_type=="valid" and is_train(img_name):
        skip_cnt+=1
        continue

    if (((db_type == 'train' or db_type == 'valid') and annot_file['img_train'][0][0][0][img_id] == 1) or (db_type == 'test' and annot_file['img_train'][0][0][0][img_id] == 0)) and \
        check_empty(annot_file['annolist'][0][0][0][img_id],'annorect') == False: #any person is annotated

        filename = img_path + img_name #filename

        img = Image.open(filename)
        w,h = img.size
        img_dict = {'id': img_id, 'file_name': "%012d.jpg" % img_id, 'width': w, 'height': h}
        coco['images'].append(img_dict)
        shutil.copy(filename, cpy_des_path+ "/"+"%012d.jpg" % img_id)

        if db_type == 'test':
            continue

        person_num = len(annot_file['annolist'][0][0][0][img_id]['annorect'][0]) #person_num
        joint_annotated = np.zeros((person_num,joint_num))
        for pid in range(person_num):
            # print("img_id", img_id)
            # print("pid", pid)
            if check_empty(annot_file['annolist'][0][0][0][img_id]['annorect'][0][pid],'annopoints') == False: #kps is annotated

                bbox = np.zeros((4)) # xmin, ymin, w, h
                kps = np.zeros((joint_num,3)) # xcoord, ycoord, vis

                #kps
                annot_joint_num = len(annot_file['annolist'][0][0][0][img_id]['annorect'][0][pid]['annopoints']['point'][0][0][0])
                for jid in range(annot_joint_num):
                    annot_jid = annot_file['annolist'][0][0][0][img_id]['annorect'][0][pid]['annopoints']['point'][0][0][0][jid]['id'][0][0]
                    kps[annot_jid][0] = annot_file['annolist'][0][0][0][img_id]['annorect'][0][pid]['annopoints']['point'][0][0][0][jid]['x'][0][0]
                    kps[annot_jid][1] = annot_file['annolist'][0][0][0][img_id]['annorect'][0][pid]['annopoints']['point'][0][0][0][jid]['y'][0][0]
                    kps[annot_jid][2] = 2

                #bbox extract from annotated kps
                annot_kps = kps[kps[:,2]==2,:].reshape(-1,3)
                # print("kps",kps)
                xmin = np.min(annot_kps[:,0])
                ymin = np.min(annot_kps[:,1])
                xmax = np.max(annot_kps[:,0])
                ymax = np.max(annot_kps[:,1])
                width = xmax - xmin - 1
                height = ymax - ymin - 1

                # corrupted bounding box
                if width <= 0 or height <= 0:
                    continue
                # 20% extend
                else:
                    bbox[0] = (xmin + xmax)/2. - width/2*1.2
                    bbox[1] = (ymin + ymax)/2. - height/2*1.2
                    bbox[2] = width*1.2
                    bbox[3] = height*1.2

                """reorder kps"""
                kps_lst = kps.reshape(-1).tolist()
                """
                "None", "pelvis", "throax","upper_neck", "head_top",
                "left_shoulder","right_shoulder","left_elbow","right_elbow",
                "left_wrist","right_wrist","left_hip","right_hip",
                "left_knee","right_knee","left_ankle","right_ankle"

                "r_ankle", "r_knee","r_hip",
                3-5 "l_hip", "l_knee", "l_ankle",
                6-9 "pelvis", "throax","upper_neck", "head_top",
                "r_wrist", "r_elbow", "r_shoulder",
                "l_shoulder", "l_elbow", "l_wrist"]
                """
                kps_reorder = [0,0,0] # None
                kps_reorder.extend(kps_lst[3*6:3*6+3]) # pelvis
                kps_reorder.extend(kps_lst[3*7:3*7+3]) # throax
                kps_reorder.extend(kps_lst[3*8:3*8+3]) # upper_neck
                kps_reorder.extend(kps_lst[3*9:3*9+3]) # head_top
                kps_reorder.extend(kps_lst[3*13:3*13+3])
                kps_reorder.extend(kps_lst[3*12:3*12+3])
                kps_reorder.extend(kps_lst[3*14:3*14+3])
                kps_reorder.extend(kps_lst[3*11:3*11+3])
                kps_reorder.extend(kps_lst[3*15:3*15+3])
                kps_reorder.extend(kps_lst[3*10:3*10+3])
                kps_reorder.extend(kps_lst[3*3:3*3+3])
                kps_reorder.extend(kps_lst[3*2:3*2+3])
                kps_reorder.extend(kps_lst[3*4:3*4+3])
                kps_reorder.extend(kps_lst[3*1:3*1+3])
                kps_reorder.extend(kps_lst[3*5:3*5+3])
                kps_reorder.extend(kps_lst[3*0:3*0+3])

                person_dict = {'id': aid, 'image_id': img_id, 'category_id': 1, 'area': bbox[2]*bbox[3], 'bbox': bbox.tolist(), 'iscrowd': 0,
                                'keypoints': kps_reorder, 'num_keypoints': int(np.sum(kps[:,2]==2))}
                coco['annotations'].append(person_dict)
                aid += 1

# category = {
#     "supercategory": "person",
#     "id": 1,  # to be same as COCO, not using 0
#     "name": "person",
#     "skeleton": [[0,1],
#         [1,2],
#         [2,6],
#         [7,12],
#         [12,11],
#         [11,10],
#         [5,4],
#         [4,3],
#         [3,6],
#         [7,13],
#         [13,14],
#         [14,15],
#         [6,7],
#         [7,8],
#         [8,9]] ,
#     "keypoints": ["r_ankle", "r_knee","r_hip",
#                     "l_hip", "l_knee", "l_ankle",
#                   "pelvis", "throax",
#                   "upper_neck", "head_top",
#                   "r_wrist", "r_elbow", "r_shoulder",
#                   "l_shoulder", "l_elbow", "l_wrist"]}

coco['categories'] = categories

with open(save_path, 'w') as f:
    json.dump(coco, f)
print("skip count:", skip_cnt)
print("Total:", img_num)
