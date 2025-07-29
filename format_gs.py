import os
import shutil
import cv2
import numpy as np
import math
import json
from tqdm import tqdm

flame_dir_path = "/mnt/nas/sitt/demo/data_prep/jea_scans/bird_blue_shirt/flame/eyes_lid"
img_dir_path = "/mnt/nas/sitt/demo/data_prep/jea_scans/bird_blue_shirt/lp_img/eyes_lid"
camera_dir_path = "/mnt/nas/sitt/demo/data_prep/lumio_scans/bird_blue_shirt/cameras"

output_dir_path = "/mnt/nas/sitt/demo/data_prep/jea_out/bbs_eyes_lid"

flame_dir = os.listdir(flame_dir_path)
img_dir = os.listdir(img_dir_path)

"""
    input flame params
    - flame_dir_path
        - jaw_0_1.npz
        - jaw_0_2.npz
        ...
        (param_timestep.npz)
        (For now, timestep start at 1. it might change idk)

    output flame params
    1. - output_dir_path/flame_param
            - 00000.npz
            - 00001.npz
            ...
            (1 per timestep)
    2. - output_dir_path
            - canonical_flame_param.npz
            (only shape param with jaw[0] = 0.3)
"""
try:
    shutil.rmtree(output_dir_path)
except:
    print("ok")

os.mkdir(output_dir_path)
os.mkdir(output_dir_path + "/images")
os.mkdir(output_dir_path + "/flame_param")
os.mkdir(output_dir_path + "/fg_masks")

for flame_name in tqdm(flame_dir, desc="Processing flame params"):

    if not flame_name.endswith(".npz"):
        continue

    s = flame_name.split(".")[0].split("_")
    # jaw_0_60_33 . npz
    pose = "0" * (2 - len(s[1])) + s[1]
    time = str(int(s[1]) - 1)
    time = "0" * (5 - len(time)) + time
    # if pose != "00":
    #     continue
    rename = time + ".npz"
    cmd = (
        "cp "
        + flame_dir_path
        + "/"
        + flame_name
        + " "
        + output_dir_path
        + "/flame_param/"
        + rename
    )
    # print(cmd)
    os.system(cmd)

flame_data = np.load(flame_dir_path + "/" + flame_dir[0])

tran = np.zeros((1, 3))
rot = np.zeros((1, 3))
neck = np.zeros((1, 3))
jaw = np.zeros((1, 3))
eye = np.zeros((1, 6))
expr = np.zeros((1, 100))

jaw[0][0] = 0.3

np.savez(
    output_dir_path + "/canonical_flame_param.npz",
    translation=np.zeros((1, 3)),
    rotation=np.zeros((1, 3)),
    neck_pose=np.zeros((1, 3)),
    jaw_pose=np.zeros((1, 3)),
    eyes_pose=np.zeros((1, 3)),
    shape=flame_data["shape"],
    expr=np.zeros((1, 100)),
)

"""
    input images
    - img_dir_path
        - cam_1_000000.png
        - cam_1_000001.png
        ...
        (cam_id_timestep.png RGB black bg image)

    output images
    1. - output_dir_path/fg_masks
            - 00000_00.png
            - 00000_01.png
            ...
            (timestep_camID.png, RGB image, 0, 0,0 = bg)
    2. - output_dir_path/images
            - 00000_00.png
            - 00000_01.png
            ...
            (timestep_camID.png, RGBA image)
"""
for img_name in tqdm(img_dir, desc="Processing images"):
    s = img_name.split(".")[0].split("_")
    cam = "0" * (2 - len(s[1])) + s[1]
    # if cam == "01" or cam == "06":
    #     continue
    time = str(int(s[2][:]))
    time = "0" * (5 - len(time)) + time
    img = cv2.imread(img_dir_path + "/" + img_name)
    ih, iw, ic = img.shape
    alpha = np.zeros((ih, iw, 1))
    # alpha[:, :, 0] = (np.sum(img, axis=2) < 20) * 255

    alpha[:, :, 0] = (np.mean(img, axis=2) < 200) * 255

    img = np.append(img, alpha, axis=2)
    mask = np.concatenate((alpha, alpha, alpha), axis=2)
    cv2.imwrite(output_dir_path + "/fg_masks/" + time + "_" + cam + ".png", mask)
    cv2.imwrite(output_dir_path + "/images/" + time + "_" + cam + ".png", img)
    # break for debug

"""
    input camera params
    1. - camera_dir_path
            - camera00.txt
            - camrea01.txt
            ...
            (lumio's camera params)
    2. camera_ids (python's list)
    3. timesteps (0-?)
    4. flip (python set) for some images set

    output camera params
    1. transforms.json
    2 - 5. transforms.json copys
    (can modified to choose test, train, validate timestep and cam)
"""

# flip = {3}
flip = {3, 4, 5, 6, 7, 10, 11, 12, 13, 14}  # shu
# flip = {0,1,2,4,6,8,9,11,15} #girl1
# camera_ids = [i for i in range(1,7)]
camera_ids = [4]
timestep = 120
flip2 = {}

transforms = {"camera_indices": camera_ids}
transforms["timestep_indices"] = list(range(timestep))
camera_params = {}
w2c = {}
intrinsics = {}

rot_90 = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

rot_180 = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

mirror = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

to_gs = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

frames = []

for time in tqdm(range(timestep), desc="Processing frames"):
    ntime = str(time)
    ntime = "0" * (5 - len(ntime)) + ntime
    for i in camera_ids:
        cam = str(i)
        cam = "0" * (2 - len(cam)) + cam
        now_frame = {
            "timestep_index": time,
            "timestep_index_original": time,
            "timestep_id": "frame_" + ntime,
            "camera_index": i,
            "camera_id": "camera" + cam,
            "file_path": "images/" + ntime + "_" + cam + ".png",
            "fg_mask_path": "fg_masks/" + ntime + "_" + cam + ".png",
            "flame_param_path": "flame_param/" + ntime + ".npz",
        }
        f = open(camera_dir_path + "/camera" + cam + ".txt")
        s = f.read()
        s = s.split("\n")
        # scale = 0.4
        scale = 1.0
        h, w = [float(i) * scale for i in s[1].split()]
        # h, w       = [972.0, 736.0]
        fl_y, fl_x = [float(i) * scale for i in s[3].split()]
        cy, cx = [float(i) * scale for i in s[5].split()]
        # cy, cx     = [972.0 / 2.0, 736.0 / 2.0]
        m = np.array([[float(j) for j in i.split()] for i in s[12:16]])
        sh = 574.0 / h
        sw = 434.0 / w
        w *= sw
        fl_x *= sw
        cx *= sw
        h *= sh
        fl_y *= sh
        cy *= sh
        intr = np.array(
            [
                [fl_x, 0.0, cx],
                [0.0, fl_y, cy],
                [0.0, 0.0, 1.0],
            ]
        )

        r = np.identity(4)
        t = np.identity(4)
        r[:3, :3] = m[:3, :3]
        t[:3, 3] = m[:3, 3] * -1.0 * 0.01
        r = r @ rot_90
        if i in flip:
            r = r @ rot_180
        if i in flip2:
            r = r @ rot_180
        r = r @ mirror
        r = r.T
        r2 = r.copy()
        r2 = r2.T
        r2 = r2 @ to_gs
        t2 = t.copy()
        t2[:3, 3] *= -1.0
        m = r @ t
        m2 = t2 @ r2
        angle_x = math.atan(w / (fl_x * 2)) * 2
        angle_y = math.atan(h / (fl_y * 2)) * 2

        now_frame["transform_matrix"] = m2.tolist()
        now_frame["cx"] = cx
        now_frame["cy"] = cy
        now_frame["fl_x"] = fl_x
        now_frame["fl_y"] = fl_y
        now_frame["h"] = h
        now_frame["w"] = w
        now_frame["camera_angle_x"] = angle_x
        now_frame["camera_angle_y"] = angle_y

        frames.append(now_frame)

transforms["frames"] = frames
transforms["cx"] = transforms["frames"][0]["cx"]
transforms["cy"] = transforms["frames"][0]["cy"]
transforms["fl_x"] = transforms["frames"][0]["fl_x"]
transforms["fl_y"] = transforms["frames"][0]["fl_y"]
transforms["h"] = transforms["frames"][0]["h"]
transforms["w"] = transforms["frames"][0]["w"]
transforms["camera_angle_x"] = transforms["frames"][0]["camera_angle_x"]
transforms["camera_angle_y"] = transforms["frames"][0]["camera_angle_y"]

json_object = json.dumps(transforms, indent=4)

with open(output_dir_path + "/transforms.json", "x") as outfile:
    outfile.write(json_object)
with open(output_dir_path + "/transforms_backup_flame.json", "x") as outfile:
    outfile.write(json_object)
with open(output_dir_path + "/transforms_backup.json", "x") as outfile:
    outfile.write(json_object)
with open(output_dir_path + "/transforms_val.json", "x") as outfile:
    outfile.write(json_object)
with open(output_dir_path + "/transforms_train.json", "x") as outfile:
    outfile.write(json_object)
with open(output_dir_path + "/transforms_test.json", "x") as outfile:
    outfile.write(json_object)
