import numpy as np


path = "/mnt/nas/sitt/demo/data_prep/motion.npz"


motion = np.load(path)

print(motion.files)

# shapes 
# ['shape', 'expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'translation', 'static_offset', 'dynamic_offset']

for key in motion.files:
    print(f"{key}: {motion[key].shape}")

# shape: (300,)
# expr: (1119, 100)
# rotation: (1119, 3)
# neck_pose: (1119, 3)
# jaw_pose: (1119, 3)
# eyes_pose: (1119, 6)
# translation: (1119, 3)
# static_offset: (1, 5143, 3)
# dynamic_offset: (1119, 5143, 3)

def select_intervals(motion, start, end):

    selected_motion = {
        key: motion[key][start:end] if key not in ['static_offset', 'shape'] else motion[key]
        for key in motion.files
    }

    return selected_motion


selected_motion = select_intervals(motion, 400, 1119)

# savez
np.savez_compressed("selected_motion.npz", **selected_motion)