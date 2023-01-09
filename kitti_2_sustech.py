
# detection ./data/example/label ./data/example/calib ./data/example/tgt_label
# calib 坐标系
# camera 相机坐标系
# label 采用kitti格式的标注数据
# lidar 激光雷达数据
# tgt_label
import os
import json
import math
import numpy as np
import sys


def get_inv_matrix(file, v2c, rect):
    with open(file) as f:
        lines = f.readlines()
        trans = [x for x in filter(lambda s: s.startswith(v2c), lines)][0]

        matrix = [m for m in map(lambda x: float(x), trans.strip().split(" ")[1:])]
        matrix = matrix + [0, 0, 0, 1]
        m = np.array(matrix)
        velo_to_cam = m.reshape([4, 4])

        trans = [x for x in filter(lambda s: s.startswith(rect), lines)][0]
        matrix = [m for m in map(lambda x: float(x), trans.strip().split(" ")[1:])]
        m = np.array(matrix).reshape(3, 3)

        m = np.concatenate((m, np.expand_dims(np.zeros(3), 1)), axis=1)

        rect = np.concatenate((m, np.expand_dims(np.array([0, 0, 0, 1]), 0)), axis=0)

        print(velo_to_cam, rect)
        m = np.matmul(rect, velo_to_cam)

        m = np.linalg.inv(m)

        return m

def get_detection_inv_matrix(calib_path, frame):
    file = os.path.join(calib_path, frame+".txt")
    return get_inv_matrix(file, "Tr_velo_to_cam", "R0_rect")

def parse_one_detection_obj(inv_matrix, l):
    words = l.strip().split(" ")
    obj = {}

    pos = np.array([float(words[11]), float(words[12]), float(words[13]), 1]).T
    trans_pos = np.matmul(inv_matrix, pos)
    #print(trans_pos)

    obj["obj_type"] = words[0]
    obj["psr"] = {"scale":
                {"z":float(words[8]),    #height
                    "x":float(words[10]),  #length
                    "y":float(words[9])},  #width
                    "position": {"x":trans_pos[0], "y":trans_pos[1], "z":trans_pos[2]+float(words[8])/2},
                    "rotation": {"x":0,
                                "y":0,
                                #"z": +math.pi/2 +float(words[14])}}
                                "z": -math.pi/2 -float(words[14])}}
    obj["obj_id"] = ""
    return obj

def trans_detection_label(src_label_path, src_calib_path, tgt_label_path):
    files = os.listdir(src_label_path)
    files.sort()

    for fname in files:
        frame, _ = os.path.splitext(fname)
        print(frame)

        inv_m = get_detection_inv_matrix(src_calib_path,frame)

        print('xxx')

        with open(os.path.join(src_label_path,fname)) as f:
            lines = f.readlines()
            objs = map(lambda l:parse_one_detection_obj(inv_m,l),lines)
            filtered_objs = [x for x in objs]
            with open(os.path.join(tgt_label_path,frame+".json"),'w') as outfile:
                json.dump(filtered_objs,outfile)
            print("compled")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("args: <detection|tracking> src_label_path src_calib_path tgt_label_path")
    else:
        label_type = sys.argv[1]
        src_label = sys.argv[2]
        src_calib = sys.argv[3]
        tgt_label = sys.argv[4]

    if label_type == "detection":
        trans_detection_label(src_label, src_calib, tgt_label)
    else:
        print("args: <detection|tracking> src_label_path src_calib_path tgt_label_path")